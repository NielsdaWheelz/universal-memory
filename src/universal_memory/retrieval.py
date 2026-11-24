from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import cast
from uuid import UUID

from pydantic import JsonValue, TypeAdapter
from sqlalchemy import and_, func, literal, literal_column, or_, select, union_all
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.sql import Select
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.selectable import ScalarSelect, Subquery

from .policy import (
    ARCHIVE_GENERATION,
    CURSOR_CHARACTERS,
    EXCHANGE_BYTES,
    FUSION_K,
    NOTICE,
    OPEN_IDENTITIES,
    POSITION_NOTICE,
    PREVIEW_BYTES,
    RECENT_EXCHANGES,
    RESULT_BYTES,
    RETAINED_TEXT_WHITESPACE,
    SEARCH_CANDIDATES,
    MemoryError,
    canonical_json,
    clip_utf8,
    encoded_size,
    validate_embedding,
)
from .schema import (
    RECORD_TABLES,
    embedding_text,
    memory_leaf,
    memory_log,
    memory_retired_source_record,
    memory_summary,
    nonblank_text,
    source_conversation,
    source_record,
)
from .types import (
    EventIdentity,
    Excerpt,
    Exchange,
    Fragment,
    Missing,
    NoteProducer,
    OpenInput,
    OpenPage,
    Provenance,
    Record,
    RecordReference,
    SearchInput,
    SearchResult,
    Store,
    Submission,
    Synthesis,
    ZoomResult,
    build_record,
)

type ResultSize = Callable[[dict[str, JsonValue]], int]
type _OwnerSpan = tuple[UUID, int, int]
_NOTE_PRODUCER: TypeAdapter[NoteProducer] = TypeAdapter(NoteProducer)


_HEAD = source_record.alias("event_head")
# Archive admission assigns an event's parts consecutive source and leaf positions
# in one transaction. The first leaf and its count therefore prove the full cut.
_EVENT_HEADS = (
    select(
        _HEAD.c.id,
        _HEAD.c.conversation_id,
        _HEAD.c.sequence,
        _HEAD.c.native_event_id,
        _HEAD.c.part_count,
        _HEAD.c.turn_id,
        _HEAD.c.native_parent_id,
        _HEAD.c.role,
        _HEAD.c.kind,
        _HEAD.c.attributes["inherited_origin"].astext.is_not(None).label("inherited"),
        _HEAD.c.attributes["reason"].astext.label("reason"),
        _HEAD.c.occurred_at,
        _HEAD.c.received_at,
        memory_leaf.c.position,
        source_conversation.c.provider,
        source_conversation.c.machine,
        source_conversation.c.account,
    )
    .join(memory_leaf, memory_leaf.c.source_record_id == _HEAD.c.id)
    .join(source_conversation, source_conversation.c.id == _HEAD.c.conversation_id)
    .where(_HEAD.c.part == literal_column("0"))
)
_HAS_TEXT = (
    select(1)
    .where(
        source_record.c.conversation_id == _HEAD.c.conversation_id,
        source_record.c.native_event_id == _HEAD.c.native_event_id,
        nonblank_text(source_record.c.text),
    )
    .exists()
)
_MECHANICAL = {"tool_call", "tool_result", "attachment", "context", "memory_reference"}


def _same_owner_message(
    anchor: RowMapping, next_row: RowMapping, lineage: dict[str, str]
) -> bool:
    # The native codecs can emit opaque ids for every block. Only the stored
    # role, conversation order, clock and lineage can identify a shared message.
    return (
        anchor["provider"] in {"codex", "claude"}
        and (
            next_row["role"] == "owner"
            or (next_row["role"] == "unknown" and next_row["kind"] == "context")
        )
        and next_row["kind"] in {"text", "attachment", "context", "memory_reference"}
        and anchor["occurred_at"] == next_row["occurred_at"]
        and not any(
            next_row[key] is not None and next_row[key] != expected
            for key, expected in lineage.items()
        )
    )


async def _exchange_side(
    connection: AsyncConnection,
    anchor: RowMapping,
    cutoff: int,
    excluded: Sequence[tuple[int, int]],
    lineage: dict[str, str],
    *,
    before: bool,
) -> tuple[list[RowMapping], RowMapping | None]:
    sequence = _HEAD.c.sequence
    statement = (
        _EVENT_HEADS.add_columns(_HAS_TEXT.label("has_text"))
        .where(
            _HEAD.c.conversation_id == anchor["conversation_id"],
            sequence < anchor["sequence"] if before else sequence > anchor["sequence"],
            memory_leaf.c.position + _HEAD.c.part_count <= cutoff,
        )
        .order_by(sequence.desc() if before else sequence)
    )
    owner: list[RowMapping] = []
    inside_unit = True
    async with connection.stream(statement) as stream:
        async for row in stream.mappings().yield_per(24):
            if (
                any(start <= row["sequence"] <= end for start, end in excluded)
                or row["inherited"]
                or row["kind"] == "revision"
                or (row["kind"] == "gap" and row["reason"] != "unsupported")
            ):
                break
            if inside_unit and _same_owner_message(anchor, row, lineage):
                if row["role"] == "owner" and row["kind"] == "text":
                    owner.append(row)
                for key in ("turn_id", "native_parent_id"):
                    if row[key] is not None:
                        lineage[key] = row[key]
                continue
            inside_unit = False
            if row["kind"] in _MECHANICAL or row["kind"] == "gap":
                continue
            if row["role"] not in {"assistant", "peer"}:
                break
            if row["has_text"]:
                return owner, row
    return owner, None


@dataclass(frozen=True, slots=True)
class _ExcerptSource:
    row: RowMapping
    head: RowMapping
    tail: RowMapping
    size: int

    def excerpt(self, allowance: int) -> Excerpt:
        head = (
            self.head["text"].encode()[: (allowance + 1) // 2].decode(errors="ignore")
        )
        tail = self.tail["text"].encode()[-(allowance // 2) :].decode(errors="ignore")
        left = cast(int, self.head["offset"])
        right = cast(int, self.tail["offset"]) - len(tail.encode())
        fragments: list[Fragment] = []
        portions = [(self.head, left, head), (self.tail, right, tail)]
        if self.head["part"] == self.tail["part"]:
            portions.sort(key=lambda part: part[1])
            first, second = portions
            shared = first[1] + len(first[2].encode()) - second[1]
            if shared >= 0:
                portions = [
                    (
                        first[0],
                        first[1],
                        first[2] + second[2].encode()[shared:].decode(),
                    )
                ]
        for row, offset, retained in portions:
            if retained:
                fragments.append(
                    Fragment(
                        ref=RecordReference(store="source_record", id=row["id"]),
                        part=row["part"],
                        byte_offset=offset,
                        text=retained,
                    )
                )
        row = self.row
        return Excerpt(
            event=EventIdentity(
                native_event_id=row["native_event_id"],
                conversation_id=row["conversation_id"],
            ),
            first_position=row["position"],
            part_count=row["part_count"],
            role=row["role"],
            provider=row["provider"],
            machine=row["machine"],
            account=row["account"],
            occurred_at=row["occurred_at"],
            received_at=row["received_at"],
            fragments=tuple(fragments),
            omitted_bytes=self.size
            - sum(len(item.text.encode()) for item in fragments),
        )


async def _excerpt_source(
    connection: AsyncConnection, row: RowMapping
) -> _ExcerptSource:
    source = source_record.c
    event = (
        source.conversation_id == row["conversation_id"],
        source.native_event_id == row["native_event_id"],
    )
    size = func.octet_length(source.text)
    total = select(func.sum(size)).where(*event).scalar_subquery()
    # Preserve all whitespace in a small part when it fits. For long parts,
    # read the first and last useful slices rather than a multipart paste.
    leading = func.ltrim(source.text, RETAINED_TEXT_WHITESPACE)
    trailing = func.rtrim(source.text, RETAINED_TEXT_WHITESPACE)
    head = (
        select(
            source.id,
            source.part,
            func.left(leading, EXCHANGE_BYTES).label("text"),
            (size - func.octet_length(leading)).label("offset"),
            total.label("size"),
        )
        .where(*event, nonblank_text(source.text))
        .order_by(source.part)
        .limit(1)
    )
    tail = (
        select(
            source.id,
            source.part,
            func.right(trailing, EXCHANGE_BYTES).label("text"),
            func.octet_length(trailing).label("offset"),
            total.label("size"),
        )
        .where(*event, nonblank_text(source.text))
        .order_by(source.part.desc())
        .limit(1)
    )
    first, last = (await connection.execute(union_all(head, tail))).mappings().all()
    return _ExcerptSource(row, first, last, first["size"])


async def _fit_exchange(
    connection: AsyncConnection,
    anchor: RowMapping,
    owner: list[RowMapping],
    preceding: RowMapping | None,
    following: RowMapping | None,
    fits: Callable[[Exchange], bool],
) -> Exchange:
    quotes: dict[int, Excerpt] = {}
    neighbors = tuple(row for row in (preceding, following) if row is not None)

    def exchange() -> Exchange:
        owner_quotes = tuple(
            quotes[row["position"]] for row in owner if row["position"] in quotes
        )
        return Exchange(
            owner=owner_quotes,
            omitted_owner_events=len(owner) - len(owner_quotes),
            preceding=quotes.get(preceding["position"])
            if preceding is not None
            else None,
            following=quotes.get(following["position"])
            if following is not None
            else None,
        )

    # Give the anchor the first claim on text after fitting the neighboring
    # source identities. The minimum buys one code point at each visible end.
    minimum = 8
    anchor_source = await _excerpt_source(connection, anchor)
    position = cast(int, anchor["position"])
    quotes[position] = anchor_source.excerpt(minimum)
    if not fits(exchange()):
        raise MemoryError("unavailable")

    sources: dict[int, _ExcerptSource] = {position: anchor_source}

    # Fit the outside companion blocks before working inward, keeping the
    # selected anchor even when it lies in the middle of the native message.
    other = [row for row in owner if row["id"] != anchor["id"] and row["has_text"]]
    priority = [
        other[step // 2] if step % 2 == 0 else other[-1 - step // 2]
        for step in range(len(other))
    ]
    for candidates in (priority, neighbors):
        for rank, row in enumerate(candidates):
            source = await _excerpt_source(connection, row)
            key = cast(int, row["position"])
            quotes[key] = source.excerpt(minimum)
            # Keep a share for the anchor at least as large as all other
            # minimum snippets combined before accepting their metadata.
            quotes[position] = anchor_source.excerpt(minimum * len(quotes))
            if not fits(exchange()):
                del quotes[key]
                quotes[position] = anchor_source.excerpt(minimum * len(quotes))
                if candidates is priority and rank >= 1:
                    break
                continue
            sources[key] = source

    for target, source in sources.items():
        high = EXCHANGE_BYTES
        best = quotes[target]
        quotes[target] = source.excerpt(high)
        if fits(exchange()):
            continue
        low = minimum * (len(sources) if target == position else 1) + 1
        while low <= high:
            allowance = (low + high) // 2
            candidate = source.excerpt(allowance)
            quotes[target] = candidate
            if fits(exchange()):
                best = candidate
                low = allowance + 1
            else:
                high = allowance - 1
        quotes[target] = best
    return exchange()


async def recent_exchanges(
    connection: AsyncConnection, cutoff: int, fits: Callable[[Exchange], bool]
) -> tuple[Exchange, ...]:
    dated = and_(
        _HEAD.c.occurred_at.is_not(None), _HEAD.c.occurred_at <= _HEAD.c.received_at
    )
    anchors = _EVENT_HEADS.where(
        _HEAD.c.role == literal_column("'owner'"),
        _HEAD.c.kind == literal_column("'text'"),
        _HEAD.c.attributes["inherited_origin"].astext.is_(None),
        source_conversation.c.relation.is_distinct_from("child"),
        memory_leaf.c.position + _HEAD.c.part_count <= cutoff,
        _HAS_TEXT,
    )
    selected: list[tuple[RowMapping, Exchange]] = []
    uncertain: Exchange | None = None
    conversations: set[UUID] = set()
    selected_spans: list[_OwnerSpan] = []

    async def pick(*, timed: bool, distinct: bool = False) -> bool:
        nonlocal uncertain
        candidates = anchors.where(dated if timed else ~dated)
        if distinct:
            candidates = candidates.where(_HEAD.c.conversation_id.not_in(conversations))
        order = (
            (_HEAD.c.occurred_at.desc().nulls_last(),)
            if timed
            else (_HEAD.c.conversation_id.in_(conversations),)
        )
        candidates = candidates.order_by(
            *order,
            _HEAD.c.received_at.desc(),
            _HEAD.c.conversation_id,
            _HEAD.c.sequence.desc(),
        ).limit(1)
        excluded = selected_spans.copy()
        while True:
            row = (
                (
                    await connection.execute(
                        candidates.where(
                            *(
                                ~and_(
                                    _HEAD.c.conversation_id == conversation,
                                    _HEAD.c.sequence.between(start, end),
                                )
                                for conversation, start, end in excluded
                            )
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                return False
            conversation = cast(UUID, row["conversation_id"])
            local = [
                (start, end) for key, start, end in excluded if key == conversation
            ]
            lineage = {
                key: cast(str, row[key])
                for key in ("turn_id", "native_parent_id")
                if row[key] is not None
            }
            before, preceding = await _exchange_side(
                connection, row, cutoff, local, lineage, before=True
            )
            after, following = await _exchange_side(
                connection, row, cutoff, local, lineage, before=False
            )
            owner = [*reversed(before), row, *after]
            span = (conversation, owner[0]["sequence"], owner[-1]["sequence"])
            if not timed and any(
                item["has_text"]
                and item["occurred_at"] is not None
                and item["occurred_at"] <= item["received_at"]
                for item in (*before, *after)
            ):
                # Same message, split capture receipts. Its dated text ranks
                # the unit, so leave it available for the dated selection.
                excluded.append(span)
                continue
            conversations.add(conversation)
            selected_spans.append(span)
            exchange = await _fit_exchange(
                connection, row, owner, preceding, following, fits
            )
            if timed:
                selected.append((row, exchange))
            else:
                uncertain = exchange
            return True

    has_dated = await pick(timed=True)
    await pick(timed=False)
    if has_dated:
        room = RECENT_EXCHANGES - (uncertain is not None)
        for distinct in (True, False):
            while len(selected) < room and await pick(timed=True, distinct=distinct):
                pass
    # Distinct-conversation preference affects membership, not display order.
    # Stable sorts express the different directions of the four tie breakers.
    selected.sort(key=lambda unit: cast(int, unit[0]["sequence"]), reverse=True)
    selected.sort(key=lambda unit: cast(UUID, unit[0]["conversation_id"]))
    selected.sort(key=lambda unit: cast(datetime, unit[0]["received_at"]), reverse=True)
    selected.sort(key=lambda unit: cast(datetime, unit[0]["occurred_at"]), reverse=True)
    timed = [unit[1] for unit in selected]
    if uncertain is not None:
        timed.insert(min(1, len(timed)), uncertain)
    return tuple(timed)


def visible(store: Store, cutoff: int | ScalarSelect[int]) -> ColumnElement[bool]:
    if store == "source_record":
        return source_record.c.id.in_(
            select(memory_leaf.c.source_record_id).where(
                memory_leaf.c.position < cutoff
            )
        )
    if store == "memory_log":
        return or_(
            and_(
                memory_log.c.agent_submission.is_(None),
                memory_log.c.synthesis.is_(None),
            ),
            memory_log.c.id.in_(
                select(memory_leaf.c.note_id).where(memory_leaf.c.position < cutoff)
            ),
        )
    return literal(True)


def embedding_rows(cutoff: int | ScalarSelect[int]) -> Subquery:
    return union_all(
        *(
            select(
                table.c.id,
                literal(store).label("store"),
                (
                    table.c.received_at
                    if store == "source_record"
                    else table.c.created_at
                ).label("stored_at"),
                table.c.text,
                table.c.embedding.is_not(None).label("embedded"),
            ).where(visible(store, cutoff), embedding_text(store))
            for store, table in RECORD_TABLES.items()
        )
    ).subquery()


async def record(
    connection: AsyncConnection, identity: RecordReference, cutoff: int
) -> Record | None:
    if identity.store == "source_record":
        row = (
            (
                await connection.execute(
                    select(
                        source_record,
                        source_conversation.c.provider,
                        source_conversation.c.machine,
                        source_conversation.c.account,
                        source_conversation.c.native_id,
                    )
                    .join(source_conversation)
                    .where(
                        source_record.c.id == identity.id,
                        visible(identity.store, cutoff),
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        return build_record(
            store=identity.store,
            id=row["id"],
            kind=row["kind"],
            role=row["role"],
            text=row["text"],
            provenance=Provenance(
                provider=row["provider"],
                machine=row["machine"],
                account=row["account"],
                conversation_id=row["conversation_id"],
                native_conversation_id=row["native_id"],
                occurred_at=row["occurred_at"],
                received_at=row["received_at"],
            ),
            native_event_id=row["native_event_id"],
            native_digest=row["native_digest"],
            turn_id=row["turn_id"],
            native_parent_id=row["native_parent_id"],
            part=row["part"],
            part_count=row["part_count"],
            attributes=row["attributes"],
        )
    table = memory_log if identity.store == "memory_log" else memory_summary
    row = (
        (
            await connection.execute(
                select(table).where(
                    table.c.id == identity.id, visible(identity.store, cutoff)
                )
            )
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        return None
    submission = (
        _NOTE_PRODUCER.validate_json(canonical_json(row["agent_submission"]))
        if identity.store == "memory_log" and row["agent_submission"]
        else None
    )
    synthesis = (
        Synthesis.model_validate_json(canonical_json(row["synthesis"]))
        if identity.store == "memory_log" and row["synthesis"]
        else None
    )
    native = submission if isinstance(submission, Submission) else None
    return build_record(
        store=identity.store,
        id=row["id"],
        kind="note" if identity.store == "memory_log" else "summary",
        role="agent" if submission or synthesis else "unknown",
        text=row["text"],
        provenance=Provenance(
            provider=native.provider if native else None,
            machine=native.machine if native else None,
            account=native.account if native else None,
            native_conversation_id=submission.native_conversation_id
            if submission
            else None,
            created_at=row["created_at"],
            agent_submission=submission,
            synthesis=synthesis,
            legacy=submission is None and synthesis is None,
        ),
        source_memory_ids=tuple(row["source_memory_ids"])
        if identity.store == "memory_summary"
        else (),
    )


def open_page(
    records: tuple[Record | Missing, ...],
    digest: str,
    cutoff: int,
    next_offset: int,
    remaining: int,
) -> OpenPage:
    return OpenPage(
        notice=NOTICE,
        records=records,
        cursor=f"{digest}:{cutoff}:{next_offset}" if remaining else None,
        remaining=remaining,
    )


def leaf_zoom(original: Record) -> ZoomResult:
    return ZoomResult(
        notice=POSITION_NOTICE,
        record=original,
        full_event=EventIdentity(
            conversation_id=original.provenance.conversation_id,
            native_event_id=original.native_event_id,
        )
        if original.provenance.conversation_id and original.native_event_id
        else None,
    )


async def open_records(
    connection: AsyncConnection, value: OpenInput, cutoff: int, size: ResultSize
) -> OpenPage:
    selector = {
        "identities": [item.model_dump(mode="json") for item in value.identities],
        "event": value.event.model_dump(mode="json") if value.event else None,
    }
    digest = hashlib.sha256(
        canonical_json({"generation": ARCHIVE_GENERATION, **selector}).encode()
    ).hexdigest()
    offset = 0
    if value.cursor is not None:
        try:
            expected, frozen, position = value.cursor.split(":")
            frozen_cutoff, offset = int(frozen), int(position)
        except (ValueError, AttributeError):
            raise MemoryError("invalid_input") from None
        if (
            expected == hashlib.sha256(canonical_json(selector).encode()).hexdigest()
            and frozen_cutoff >= 0
            and offset >= 0
        ):
            raise MemoryError("stale_view")
        if (
            expected != digest
            or offset < 0
            or len(value.cursor) > CURSOR_CHARACTERS
            or not 0 <= frozen_cutoff <= cutoff
        ):
            raise MemoryError("invalid_input")
        cutoff = frozen_cutoff
    identities = list(value.identities)
    total = len(identities)
    if value.event is not None:
        condition = and_(
            source_record.c.conversation_id == value.event.conversation_id,
            source_record.c.native_event_id == value.event.native_event_id,
            visible("source_record", cutoff),
        )
        total = int(
            await connection.scalar(
                select(func.count()).select_from(source_record).where(condition)
            )
            or 0
        )
        rows = (
            (
                await connection.execute(
                    select(source_record.c.id)
                    .where(condition)
                    .order_by(source_record.c.part)
                    .offset(offset)
                    .limit(OPEN_IDENTITIES)
                )
            )
            .scalars()
            .all()
        )
        identities = [RecordReference(store="source_record", id=item) for item in rows]
        if total == 0:
            retired = await connection.scalar(
                select(memory_retired_source_record.c.id)
                .where(
                    memory_retired_source_record.c.conversation_id
                    == value.event.conversation_id,
                    memory_retired_source_record.c.native_event_id
                    == value.event.native_event_id,
                )
                .limit(1)
            )
            if retired is not None:
                raise MemoryError("retired")
            raise MemoryError("not_found")
    else:
        identities = identities[offset:]
    if offset > total:
        raise MemoryError("invalid_input")
    records: list[Record | Missing] = []
    for identity in identities:
        item = await record(connection, identity, cutoff)
        accepted: Record | Missing
        if item is None:
            retired = (
                identity.store == "source_record"
                and await connection.scalar(
                    select(memory_retired_source_record.c.id).where(
                        memory_retired_source_record.c.id == identity.id
                    )
                )
                is not None
            )
            accepted = Missing(
                store=identity.store,
                id=identity.id,
                error="retired" if retired else "not_found",
            )
        else:
            accepted = item
        remaining = total - offset - len(records) - 1
        page = open_page(
            (*records, accepted), digest, cutoff, offset + len(records) + 1, remaining
        )
        if (
            max(
                encoded_size(page.model_dump(mode="json")),
                size(page.model_dump(mode="json")),
            )
            > RESULT_BYTES
        ):
            if not records:
                raise MemoryError("unavailable")
            break
        records.append(accepted)
    next_offset = offset + len(records)
    return open_page(tuple(records), digest, cutoff, next_offset, total - next_offset)


async def search(
    connection: AsyncConnection,
    value: SearchInput,
    vector: Sequence[float],
    cutoff: int,
    size: ResultSize,
) -> SearchResult:
    embedding = validate_embedding(vector)
    lexical: list[Select[tuple[object, ...]]] = []
    semantic: list[Select[tuple[object, ...]]] = []
    missing = 0
    for store in value.stores:
        table = RECORD_TABLES[store]
        if (
            value.kinds is not None
            and store != "source_record"
            and ("note" if store == "memory_log" else "summary") not in value.kinds
        ):
            continue
        condition = visible(store, cutoff)
        if value.kinds is not None and store == "source_record":
            condition = and_(condition, table.c.kind.in_(value.kinds))
        stored_at = (
            table.c.received_at if store == "source_record" else table.c.created_at
        )
        english = func.to_tsvector(literal_column("'english'::regconfig"), table.c.text)
        simple = func.to_tsvector(literal_column("'simple'::regconfig"), table.c.text)
        eq = func.websearch_to_tsquery(
            literal_column("'english'::regconfig"), value.query
        )
        sq = func.websearch_to_tsquery(
            literal_column("'simple'::regconfig"), value.query
        )
        rank = func.greatest(func.ts_rank_cd(english, eq), func.ts_rank_cd(simple, sq))
        lexical.append(
            cast(
                Select[tuple[object, ...]],
                select(
                    table.c.id,
                    literal(store).label("store"),
                    stored_at.label("stored_at"),
                    rank.label("rank"),
                ).where(condition, or_(english.op("@@")(eq), simple.op("@@")(sq))),
            )
        )
        eligible = and_(condition, embedding_text(store))
        semantic.append(
            cast(
                Select[tuple[object, ...]],
                select(
                    table.c.id,
                    literal(store).label("store"),
                    stored_at.label("stored_at"),
                    table.c.embedding.cosine_distance(embedding).label("rank"),
                ).where(eligible, table.c.embedding.is_not(None)),
            )
        )
        missing += int(
            await connection.scalar(
                select(func.count())
                .select_from(table)
                .where(eligible, table.c.embedding.is_(None))
            )
            or 0
        )
    rankings: list[list[RecordReference]] = []
    for queries, descending in ((lexical, True), (semantic, False)):
        if not queries:
            rankings.append([])
            continue
        candidates = union_all(*queries).subquery()
        ordering = candidates.c.rank.desc() if descending else candidates.c.rank.asc()
        rows = (
            (
                await connection.execute(
                    select(candidates)
                    .order_by(
                        ordering,
                        candidates.c.stored_at.desc(),
                        candidates.c.store,
                        candidates.c.id,
                    )
                    .limit(SEARCH_CANDIDATES)
                )
            )
            .mappings()
            .all()
        )
        rankings.append(
            [RecordReference(store=row["store"], id=row["id"]) for row in rows]
        )
    scores: dict[RecordReference, float] = {}
    for ranking in rankings:
        for rank, identity in enumerate(ranking, 1):
            scores[identity] = scores.get(identity, 0) + 1 / (FUSION_K + rank)
    ordered = sorted(scores, key=lambda item: (-scores[item], item.store, item.id))
    records: list[Record] = []
    omitted_bytes = 0
    for identity in ordered[: value.limit]:
        original = await record(connection, identity, cutoff)
        if original is None:
            raise MemoryError("unavailable")
        preview = clip_utf8(original.text, PREVIEW_BYTES)
        clipped = original.model_copy(
            update={"text": preview, "clipped": preview != original.text}
        )
        page = SearchResult(
            notice=NOTICE,
            records=(*records, clipped),
            lexical_candidates=len(rankings[0]),
            semantic_candidates=len(rankings[1]),
            candidates=len(scores),
            returned=len(records) + 1,
            omitted_limit=max(0, len(scores) - value.limit),
            omitted_bytes=omitted_bytes,
            missing_embeddings=missing,
        )
        if (
            max(
                encoded_size(page.model_dump(mode="json")),
                size(page.model_dump(mode="json")),
            )
            > RESULT_BYTES
        ):
            omitted_bytes += 1
            continue
        records.append(clipped)
    return SearchResult(
        notice=NOTICE,
        records=tuple(records),
        lexical_candidates=len(rankings[0]),
        semantic_candidates=len(rankings[1]),
        candidates=len(scores),
        returned=len(records),
        omitted_limit=max(0, len(scores) - value.limit),
        omitted_bytes=omitted_bytes,
        missing_embeddings=missing,
    )
