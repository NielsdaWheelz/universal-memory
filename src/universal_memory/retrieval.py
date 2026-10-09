from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import cast

from pydantic import JsonValue, TypeAdapter
from sqlalchemy import and_, func, literal, literal_column, or_, select, union_all
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.sql import Select
from sqlalchemy.sql.elements import ColumnElement

from .policy import (
    CURSOR_CHARACTERS,
    FUSION_K,
    NOTICE,
    OPEN_IDENTITIES,
    PREVIEW_BYTES,
    RESULT_BYTES,
    SEARCH_CANDIDATES,
    MemoryError,
    canonical_json,
    clip_utf8,
    encoded_size,
    validate_embedding,
)
from .schema import (
    RECORD_TABLES,
    conversational_text,
    memory_leaf,
    memory_log,
    memory_summary,
    source_conversation,
    source_record,
)
from .types import (
    EventIdentity,
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
_NOTE_PRODUCER: TypeAdapter[NoteProducer] = TypeAdapter(NoteProducer)


def visible(store: Store, cutoff: int) -> ColumnElement[bool]:
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
        notice=NOTICE,
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
    digest = hashlib.sha256(canonical_json(selector).encode()).hexdigest()
    offset = 0
    if value.cursor is not None:
        try:
            expected, frozen, position = value.cursor.split(":")
            frozen_cutoff, offset = int(frozen), int(position)
        except (ValueError, AttributeError):
            raise MemoryError("invalid_input") from None
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
            raise MemoryError("not_found")
    else:
        identities = identities[offset:]
    if offset > total:
        raise MemoryError("invalid_input")
    records: list[Record | Missing] = []
    for identity in identities:
        item = await record(connection, identity, cutoff)
        accepted = item or Missing(store=identity.store, id=identity.id)
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
        semantic.append(
            cast(
                Select[tuple[object, ...]],
                select(
                    table.c.id,
                    literal(store).label("store"),
                    stored_at.label("stored_at"),
                    table.c.embedding.cosine_distance(embedding).label("rank"),
                ).where(condition, table.c.embedding.is_not(None)),
            )
        )
        eligible = condition
        if store == "source_record":
            eligible = and_(eligible, conversational_text)
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
