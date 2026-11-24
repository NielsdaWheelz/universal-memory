from __future__ import annotations

from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4
from xml.sax.saxutils import escape

from llm_tools import prompt_json_text
from pydantic import JsonValue, ValidationError
from sqlalchemy import func, literal, select, text, tuple_, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine
from sqlalchemy.sql.elements import ColumnElement

from . import archive, dreaming, nexus, retrieval, tree
from .nexus import NexusIngest, NexusReceipt
from .policy import (
    ARCHIVE_GENERATION,
    CURSOR_CHARACTERS,
    EMBEDDING_BATCH_SIZE,
    EXCHANGE_BYTES,
    MAIN_UPPER_BYTES,
    POSITION_NOTICE,
    READ_STATEMENT_SECONDS,
    READY_VIEW_TARGET_BYTES,
    RECENT_BYTES,
    RESULT_BYTES,
    VIEW_PAGE_BYTES,
    VIEW_SNAPSHOTS,
    MemoryError,
    encoded_size,
    utf8_size,
    validate_embedding,
)
from .schema import (
    RECORD_TABLES,
    embedding_text,
    memory_lane,
    memory_leaf,
    memory_log,
    memory_node,
    memory_state,
    source_conversation,
    source_record,
)
from .types import (
    CompactionBatch,
    ConversationReceipt,
    DateResult,
    DreamInput,
    Event,
    Exchange,
    Lane,
    Node,
    Occurrence,
    OpenInput,
    OpenPage,
    ReadyView,
    RecordReference,
    SearchInput,
    SearchResult,
    Store,
    ViewPage,
    ZoomResult,
)

type EmbeddingKey = tuple[datetime, Store, UUID]


@dataclass(frozen=True, slots=True)
class EmbeddingCandidate:
    key: EmbeddingKey
    text: str

    @property
    def reference(self) -> RecordReference:
        _, store, identifier = self.key
        return RecordReference(store=store, id=identifier)


def render_ready_view(view: ReadyView) -> str:
    return (
        tree.render(view.nodes, view.through, generation=view.generation)
        + f"\nadmitted_through={view.admitted_through}. "
        + f"{view.admitted_through - view.through} leaves outside the complete prefix. "
        + "These positions do not establish capture freshness.\n"
        + "Recent exchanges quote retained historical text and may overlap the prefix. "
        + "omitted_bytes counts retained bytes left out of an excerpt. "
        + "Open the source event to check earlier capture omissions. "
        + "Terminal user-role text carries no verified authorship or present approval."
        + "".join(_render_exchange(exchange) for exchange in view.recent)
    )


def _render_exchange(exchange: Exchange) -> str:
    lines = [
        "exchange",
        f"wholly omitted owner text events={exchange.omitted_owner_events}",
    ]
    for label, excerpt in (
        ("preceding message", exchange.preceding),
        *(("owner message", item) for item in exchange.owner),
        ("following message", exchange.following),
    ):
        if excerpt is None:
            continue
        source_time = (
            "source time unknown"
            if excerpt.occurred_at is None
            else "source clock ahead"
            if excerpt.occurred_at > excerpt.received_at
            else "source time"
        )
        role = (
            "terminal user-role"
            if excerpt.role == "owner" and excerpt.provider in {"codex", "claude"}
            else excerpt.role
        )
        lines.append(
            f"{label} ({excerpt.provider} {role}, {source_time})|"
            + prompt_json_text(excerpt.model_dump(mode="json", exclude_none=True))
        )
    return "\n" + "\n".join(lines)


def render_nexus_ready_view(view: ReadyView) -> str:
    return (
        f'<shared_memory status="ready" through="{view.through}" '
        f'admitted_through="{view.admitted_through}">\n'
        + escape(render_ready_view(view))
        + "\n</shared_memory>"
    )


async def collect_ready_view(
    first: ViewPage, next_page: Callable[[str], Awaitable[ViewPage]]
) -> ReadyView:
    """Assemble one frozen page sequence, validating each bounded prefix."""
    page = first
    cursors: set[str] = set()
    nodes: list[Node] = []
    recent: list[Exchange] = []
    end = 0
    while True:
        if (
            page.generation != first.generation
            or page.through != first.through
            or page.admitted_through != first.admitted_through
            or (recent and page.nodes)
            or (page.nodes and page.nodes[0].start != end)
        ):
            raise MemoryError("unsupported")
        if page.nodes:
            last = page.nodes[-1]
            end = last.start + last.count
        if page.recent and end != first.through:
            raise MemoryError("unsupported")
        nodes.extend(page.nodes)
        recent.extend(page.recent)
        # Includes the rest of the prefix once complete. Every partial frame
        # must also fit so malformed streams cannot accumulate unbounded text.
        try:
            value = ReadyView(
                generation=first.generation,
                nodes=tuple(nodes),
                through=end,
                admitted_through=first.admitted_through,
                recent=tuple(recent),
            )
        except ValidationError:
            raise MemoryError("unsupported") from None
        if utf8_size(render_nexus_ready_view(value)) > MAIN_UPPER_BYTES:
            raise MemoryError("unsupported")
        if page.cursor is None:
            if end != first.through:
                raise MemoryError("unsupported")
            return value
        if page.cursor in cursors:
            raise MemoryError("unsupported")
        cursors.add(page.cursor)
        page = await next_page(page.cursor)


class MemoryStore:
    """One PostgreSQL memory corpus. The caller owns every write transaction."""

    activate_lane = staticmethod(archive.activate_lane)
    sync_conversations = staticmethod(archive.sync_conversations)
    resolve_baseline = staticmethod(archive.resolve_baseline)
    register_nexus_source = staticmethod(nexus.register_nexus_source)
    nexus_capture_state = staticmethod(nexus.nexus_capture_state)
    append_note = staticmethod(archive.append_note)
    recover_note = staticmethod(archive.recover_note)
    park_capture = staticmethod(archive.park_capture)
    retry_capture = staticmethod(archive.retry_capture)
    park_compaction = staticmethod(tree.park_compaction)
    retry_compression = staticmethod(tree.retry_compression)
    complete_dream = staticmethod(dreaming.complete_dream)
    claim_nightly = staticmethod(dreaming.claim_nightly)

    def __init__(
        self,
        read_engine: AsyncEngine,
        *,
        result_size: retrieval.ResultSize = encoded_size,
        view_size: tree.ViewSize = utf8_size,
    ) -> None:
        self._read_engine = read_engine
        self._result_size = result_size
        self._view_size = view_size
        self._ready = tree.Ready()
        self._snapshots: OrderedDict[UUID, ReadyView] = OrderedDict()

    async def append_occurrences(
        self,
        connection: AsyncConnection,
        lane: Lane,
        conversation_id: UUID,
        expected_checkpoint: str | None,
        checkpoint_native_digest: str | None,
        checkpoint_capture_digest: str | None,
        occurrences: Sequence[Occurrence],
        caught_up: bool,
        observed_at: datetime,
    ) -> ConversationReceipt:
        return await archive.append_occurrences(
            connection,
            lane,
            conversation_id,
            expected_checkpoint,
            checkpoint_native_digest,
            checkpoint_capture_digest,
            occurrences,
            caught_up,
            observed_at,
            self._result_size,
        )

    async def append_main_events(
        self,
        connection: AsyncConnection,
        conversation_id: UUID,
        events: Sequence[Event],
    ) -> None:
        await archive.append_main_events(
            connection, conversation_id, events, self._result_size
        )

    async def append_main_observations(
        self,
        connection: AsyncConnection,
        conversation_id: UUID,
        events: Sequence[Event],
    ) -> None:
        await archive.append_main_events(
            connection, conversation_id, events, self._result_size, stock=True
        )

    async def append_nexus_export(
        self, connection: AsyncConnection, request: NexusIngest
    ) -> NexusReceipt:
        return await nexus.append_nexus_export(connection, request, self._result_size)

    @asynccontextmanager
    async def _read(self) -> AsyncIterator[AsyncConnection]:
        try:
            async with self._read_engine.connect() as connection:
                async with connection.begin():
                    await connection.execute(text("SET TRANSACTION READ ONLY"))
                    await connection.execute(
                        text(
                            f"SET LOCAL statement_timeout = '{READ_STATEMENT_SECONDS}s'"
                        )
                    )
                    yield connection
        except SQLAlchemyError:
            raise MemoryError("unavailable") from None

    async def cutoff(self) -> int:
        async with self._read() as connection:
            return int(await connection.scalar(select(memory_state.c.leaf_count)) or 0)

    async def _cutoff(self, connection: AsyncConnection, cutoff: int | None) -> int:
        admitted = int(await connection.scalar(select(memory_state.c.leaf_count)) or 0)
        if cutoff is not None and (
            type(cutoff) is not int or not 0 <= cutoff <= admitted
        ):
            raise MemoryError("invalid_input")
        return admitted if cutoff is None else cutoff

    async def main_view(self, cutoff: int) -> tuple[Node, ...]:
        async with self._read() as connection:
            return await tree.context_view(connection, cutoff, self._view_size)

    async def next_compaction(
        self, connection: AsyncConnection, cutoff: int | None = None
    ) -> CompactionBatch | None:
        if not connection.in_transaction():
            raise MemoryError("invalid_input")
        # Only committed nodes enter the queue; a caller rollback remains retryable.
        async with self._read() as reader:
            return await self._ready.next(reader, cutoff, self._view_size)

    async def complete_compaction(
        self,
        connection: AsyncConnection,
        batch: CompactionBatch,
        results: Sequence[Node],
    ) -> tuple[Node, ...]:
        return await tree.complete_compaction(
            connection, batch, results, self._view_size
        )

    async def ready_view(self, cutoff: int | None = None) -> ReadyView:
        async with self._read() as connection:
            admitted = await self._cutoff(connection, cutoff)
            empty = ReadyView(
                generation=ARCHIVE_GENERATION,
                nodes=(),
                through=0,
                admitted_through=admitted,
                recent=(),
            )
            empty_native = self._view_size(render_ready_view(empty))
            empty_nexus = utf8_size(render_nexus_ready_view(empty))

            def exchange_size(recent: tuple[Exchange, ...]) -> int:
                view = empty.model_copy(update={"recent": recent})
                return max(
                    self._view_size(render_ready_view(view)) - empty_native,
                    utf8_size(render_nexus_ready_view(view)) - empty_nexus,
                )

            def fits(exchange: Exchange) -> bool:
                return (
                    exchange_size((exchange,)) <= EXCHANGE_BYTES
                    and self._page_size(
                        ViewPage(
                            notice=POSITION_NOTICE,
                            nodes=(),
                            through=admitted,
                            admitted_through=admitted,
                            recent=(exchange,),
                            cursor=f"{'f' * 36}:{2**63 - 1}",
                        )
                    )
                    <= VIEW_PAGE_BYTES
                )

            recent = await retrieval.recent_exchanges(connection, admitted, fits)
            if exchange_size(recent) > RECENT_BYTES:
                raise MemoryError("unavailable")
            state = (await connection.execute(select(memory_state))).mappings().one()
            through = min(admitted, state["view_through"])
            full = await tree.load_frontier(
                connection, tree.frontier(state["main_parts"]), state["view_through"]
            )
            nodes = await tree.clip_nodes(connection, full, 0, through)

            def value(items: Sequence[Node]) -> ReadyView:
                return ReadyView(
                    generation=ARCHIVE_GENERATION,
                    nodes=tuple(items),
                    through=items[-1].start + items[-1].count if items else 0,
                    admitted_through=admitted,
                    recent=recent,
                )

            def size(view: ReadyView) -> int:
                return max(
                    self._view_size(render_ready_view(view)),
                    utf8_size(render_nexus_ready_view(view)),
                )

            best: ReadyView | None = None
            best_size = MAIN_UPPER_BYTES + 1
            while True:
                view = value(nodes)
                measured = size(view)
                if measured <= READY_VIEW_TARGET_BYTES:
                    return view
                if measured < best_size:
                    best, best_size = view, measured
                if not await tree.merge_next(connection, nodes, through):
                    break
            if best is not None:
                return best

            fitting: list[Node] = []
            for item in nodes:
                if size(value((*fitting, item))) > MAIN_UPPER_BYTES:
                    break
                fitting.append(item)
            return value(fitting)

    def _page_size(self, page: ViewPage) -> int:
        wire = page.model_dump(mode="json")
        return max(encoded_size(wire), self._result_size(wire))

    async def view(
        self, cursor: str | None = None, cutoff: int | None = None
    ) -> ViewPage:
        offset = 0
        if cursor is None:
            snapshot = await self.ready_view(cutoff)
            identifier = uuid4()
            self._snapshots[identifier] = snapshot
            if len(self._snapshots) > VIEW_SNAPSHOTS:
                self._snapshots.popitem(last=False)
        else:
            try:
                identifier_text, offset_text = cursor.split(":")
                identifier, offset = UUID(identifier_text), int(offset_text)
            except (ValueError, AttributeError):
                raise MemoryError("stale_view") from None
            snapshot = self._snapshots.get(identifier)
            if (
                snapshot is None
                or not 0 < offset < len(snapshot.nodes) + len(snapshot.recent)
                or len(cursor) > CURSOR_CHARACTERS
            ):
                raise MemoryError("stale_view")
            if cutoff is not None and snapshot.admitted_through > cutoff:
                raise MemoryError("invalid_input")
        items = (*snapshot.nodes, *snapshot.recent)

        def page(stop: int) -> ViewPage:
            selected = items[offset:stop]
            return ViewPage(
                notice=POSITION_NOTICE,
                nodes=tuple(item for item in selected if isinstance(item, Node)),
                through=snapshot.through,
                admitted_through=snapshot.admitted_through,
                recent=tuple(item for item in selected if isinstance(item, Exchange)),
                cursor=f"{identifier}:{stop}" if stop < len(items) else None,
            )

        stop = offset
        for next_stop in range(offset + 1, len(items) + 1):
            if self._page_size(page(next_stop)) > VIEW_PAGE_BYTES:
                break
            stop = next_stop
        if offset < len(items) and stop == offset:
            raise MemoryError("unavailable")
        return page(stop)

    async def zoom(
        self, start: int, count: int, cutoff: int | None = None, *, generation: int = 1
    ) -> ZoomResult:
        if generation != ARCHIVE_GENERATION:
            raise MemoryError("retired_range")
        coordinates = tree.key(start, count)
        async with self._read() as connection:
            admitted = await self._cutoff(connection, cutoff)
            if start + count > admitted:
                raise MemoryError("invalid_input")
            completed = await connection.scalar(
                select(memory_node.c.text).where(
                    memory_node.c.level == coordinates[0],
                    memory_node.c.node_index == coordinates[1],
                )
            )
            if completed is None:
                raise MemoryError("unavailable")
            if count > 1:
                half = count // 2
                children = (tree.key(start, half), tree.key(start + half, half))
                values = await tree.texts(connection, children)
                if len(values) != 2:
                    raise MemoryError("unavailable")
                result = ZoomResult(
                    notice=POSITION_NOTICE,
                    nodes=tuple(tree.node(child, values[child]) for child in children),
                )
            else:
                leaf = (
                    (
                        await connection.execute(
                            select(memory_leaf).where(memory_leaf.c.position == start)
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if leaf is None:
                    raise MemoryError("not_found")
                identity = RecordReference(
                    store="source_record" if leaf["source_record_id"] else "memory_log",
                    id=leaf["source_record_id"] or leaf["note_id"],
                )
                original = await retrieval.record(connection, identity, admitted)
                if original is None:
                    raise MemoryError("not_found")
                result = retrieval.leaf_zoom(original)
            if (
                max(
                    encoded_size(result.model_dump(mode="json")),
                    self._result_size(result.model_dump(mode="json")),
                )
                > RESULT_BYTES
            ):
                raise MemoryError("unavailable")
            return result

    async def date(
        self, position: int, cutoff: int | None = None, *, generation: int = 1
    ) -> DateResult:
        if generation != ARCHIVE_GENERATION:
            raise MemoryError("retired_range")
        async with self._read() as connection:
            admitted = await self._cutoff(connection, cutoff)
            if type(position) is not int or not 0 <= position < admitted:
                raise MemoryError("invalid_input")
            row = (
                (
                    await connection.execute(
                        select(
                            memory_leaf.c.note_id,
                            source_record.c.occurred_at,
                            source_record.c.received_at,
                            memory_log.c.created_at,
                        )
                        .outerjoin(
                            source_record,
                            memory_leaf.c.source_record_id == source_record.c.id,
                        )
                        .outerjoin(memory_log, memory_leaf.c.note_id == memory_log.c.id)
                        .where(memory_leaf.c.position == position)
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise MemoryError("not_found")
            return DateResult(
                notice=POSITION_NOTICE,
                position=position,
                occurred_at=row["occurred_at"],
                received_at=row["created_at"] if row["note_id"] else row["received_at"],
                note_created_at=row["note_id"] is not None,
            )

    async def open(self, value: OpenInput, cutoff: int | None = None) -> OpenPage:
        async with self._read() as connection:
            return await retrieval.open_records(
                connection,
                value,
                await self._cutoff(connection, cutoff),
                self._result_size,
            )

    async def search(
        self,
        value: SearchInput,
        query_vector: Sequence[float],
        cutoff: int | None = None,
    ) -> SearchResult:
        async with self._read() as connection:
            await connection.execute(
                text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            )
            return await retrieval.search(
                connection,
                value,
                query_vector,
                await self._cutoff(connection, cutoff),
                self._result_size,
            )

    async def dream_input(self, cutoff: int | None = None) -> DreamInput:
        async with self._read() as connection:
            return await dreaming.dream_input(
                connection, await self._cutoff(connection, cutoff), self._view_size
            )

    async def rebuild(
        self,
        connection: AsyncConnection,
        start: int | None = None,
        count: int | None = None,
        *,
        generation: int = 1,
    ) -> None:
        if start is not None and generation != ARCHIVE_GENERATION:
            raise MemoryError("retired_range")
        await tree.rebuild(connection, start, count, self._view_size)
        self._ready = tree.Ready()
        self._snapshots.clear()

    async def embedding_candidates(
        self,
        limit: int,
        *,
        after: EmbeddingKey | None = None,
        through: EmbeddingKey | None = None,
    ) -> tuple[EmbeddingCandidate, ...]:
        if type(limit) is not int or not 1 <= limit <= EMBEDDING_BATCH_SIZE:
            raise MemoryError("invalid_input")
        async with self._read() as connection:
            cutoff = await self._cutoff(connection, None)
            rows = retrieval.embedding_rows(cutoff).c
            columns = (
                rows.stored_at,
                cast(ColumnElement[str], rows.store.collate("C")),
                rows.id,
            )
            key = tuple_(*columns)
            query = select(*columns, rows.text).where(~rows.embedded)
            if after is not None:
                query = query.where(key > tuple_(*(literal(part) for part in after)))
            if through is not None:
                query = query.where(key <= tuple_(*(literal(part) for part in through)))
            return tuple(
                EmbeddingCandidate((stored_at, store, identifier), body)
                for stored_at, store, identifier, body in await connection.execute(
                    query.order_by(*columns).limit(limit)
                )
            )

    async def last_pending_embedding(self) -> EmbeddingKey | None:
        async with self._read() as connection:
            cutoff = await self._cutoff(connection, None)
            rows = retrieval.embedding_rows(cutoff).c
            columns = (
                rows.stored_at,
                cast(ColumnElement[str], rows.store.collate("C")),
                rows.id,
            )
            row = (
                await connection.execute(
                    select(*columns)
                    .where(~rows.embedded)
                    .order_by(*(column.desc() for column in columns))
                    .limit(1)
                )
            ).first()
            return None if row is None else (row[0], row[1], row[2])

    async def write_embeddings(
        self,
        connection: AsyncConnection,
        values: Sequence[tuple[RecordReference, Sequence[float]]],
    ) -> int:
        if not connection.in_transaction():
            raise MemoryError("invalid_input")
        if len({(ref.store, ref.id) for ref, _ in values}) != len(values):
            raise MemoryError("invalid_input")
        prepared = tuple(
            (reference, validate_embedding(vector)) for reference, vector in values
        )
        cutoff = await self._cutoff(connection, None)
        for reference, vector in prepared:
            table = RECORD_TABLES[reference.store]
            identifier = await connection.scalar(
                update(table)
                .where(
                    table.c.id == reference.id,
                    table.c.embedding.is_(None),
                    retrieval.visible(reference.store, cutoff),
                    embedding_text(reference.store),
                )
                .values(embedding=vector)
                .returning(table.c.id)
            )
            if identifier is None:
                raise MemoryError("not_found")
        return len(prepared)

    async def status(self) -> dict[str, JsonValue]:
        async with self._read() as connection:
            state = dict(
                (await connection.execute(select(memory_state))).mappings().one()
            )
            total = cast(int, state["leaf_count"])
            built = int(
                await connection.scalar(select(func.count()).select_from(memory_node))
                or 0
            )
            eligible = int(
                await connection.scalar(
                    select(func.count())
                    .select_from(
                        memory_leaf.outerjoin(
                            memory_log, memory_leaf.c.note_id == memory_log.c.id
                        )
                    )
                    .where(
                        memory_leaf.c.position >= state["dream_through"],
                        memory_log.c.synthesis.is_(None),
                    )
                )
                or 0
            )
            gaps = int(
                await connection.scalar(
                    select(func.count())
                    .select_from(source_record)
                    .where(source_record.c.kind == "gap")
                )
                or 0
            )
            lanes = (await connection.execute(select(memory_lane))).mappings().all()
            conversations = (
                (
                    await connection.execute(
                        select(
                            source_conversation.c.id,
                            source_conversation.c.provider,
                            source_conversation.c.native_id,
                            source_conversation.c.machine,
                            source_conversation.c.account,
                            source_conversation.c.capture_id,
                            source_conversation.c.capture_after_event_id,
                            source_conversation.c.baseline_error,
                            source_conversation.c.checkpoint_event_id,
                            source_conversation.c.checkpoint_native_digest,
                            source_conversation.c.checkpoint_capture_digest,
                            source_conversation.c.last_read_at,
                            source_conversation.c.capture_error,
                            source_conversation.c.worker_excluded,
                        ).order_by(
                            source_conversation.c.provider,
                            source_conversation.c.native_id,
                        )
                    )
                )
                .mappings()
                .all()
            )
            embedding_rows = retrieval.embedding_rows(
                select(memory_state.c.leaf_count).scalar_subquery()
            )
            rows = embedding_rows.c
            embedding_eligible, embedded, oldest = (
                await connection.execute(
                    select(
                        func.count(),
                        func.count().filter(rows.embedded),
                        func.min(rows.stored_at).filter(~rows.embedded),
                    ).select_from(embedding_rows)
                )
            ).one()

            def json_row(value: dict[str, object]) -> dict[str, JsonValue]:
                return {
                    name: str(item)
                    if isinstance(item, UUID | datetime)
                    else cast(JsonValue, item)
                    for name, item in value.items()
                }

            return {
                "compression": {
                    "leaf_count": total,
                    "view_through": state["view_through"],
                    "missing_nodes": 2 * total - total.bit_count() - built,
                    "compaction_error": cast(JsonValue, state["compaction_error"]),
                },
                "dreaming": {
                    "dream_through": state["dream_through"],
                    "eligible_backlog": eligible,
                    "last_nightly_attempt_at": str(state["last_nightly_attempt_at"])
                    if state["last_nightly_attempt_at"]
                    else None,
                },
                "embedding": {
                    "eligible": embedding_eligible,
                    "embedded": embedded,
                    "pending": embedding_eligible - embedded,
                    "oldest_pending_at": (
                        None if oldest is None else oldest.astimezone(UTC).isoformat()
                    ),
                },
                "gaps": gaps,
                "pending_baselines": sum(
                    row["baseline_error"] is not None and not row["worker_excluded"]
                    for row in conversations
                ),
                "excluded_workers": sum(
                    bool(row["worker_excluded"]) for row in conversations
                ),
                "lanes": [json_row(dict(row)) for row in lanes],
                "conversations": [json_row(dict(row)) for row in conversations],
            }
