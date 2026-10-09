from __future__ import annotations

from collections import OrderedDict
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import cast
from uuid import UUID, uuid4

from pydantic import JsonValue
from sqlalchemy import and_, func, select, text, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from . import archive, dreaming, retrieval, tree
from .policy import (
    CURSOR_CHARACTERS,
    MAIN_UPPER_BYTES,
    NOTICE,
    READ_STATEMENT_SECONDS,
    RESULT_BYTES,
    VIEW_SNAPSHOTS,
    MemoryError,
    encoded_size,
    utf8_size,
    validate_embedding,
)
from .schema import (
    RECORD_TABLES,
    conversational_text,
    memory_lane,
    memory_leaf,
    memory_log,
    memory_node,
    memory_state,
    source_conversation,
    source_record,
)
from .types import (
    CompactionTask,
    ConversationReceipt,
    DateResult,
    DreamInput,
    Event,
    Node,
    OpenInput,
    OpenPage,
    Record,
    RecordReference,
    SearchInput,
    SearchResult,
    ViewPage,
    ZoomResult,
)


@dataclass(frozen=True)
class _Snapshot:
    nodes: tuple[Node, ...]
    through: int
    admitted_through: int


class MemoryStore:
    """One PostgreSQL memory corpus. The caller owns every write transaction."""

    activate_lane = staticmethod(archive.activate_lane)
    sync_conversations = staticmethod(archive.sync_conversations)
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
        self._snapshots: OrderedDict[UUID, _Snapshot] = OrderedDict()

    async def append_events(
        self,
        connection: AsyncConnection,
        conversation_id: UUID,
        expected_checkpoint: str | None,
        checkpoint_native_digest: str | None,
        events: Sequence[Event],
        caught_up: bool,
        observed_at: datetime,
    ) -> ConversationReceipt:
        return await archive.append_events(
            connection,
            conversation_id,
            expected_checkpoint,
            checkpoint_native_digest,
            events,
            caught_up,
            observed_at,
            self._result_size,
        )

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
    ) -> CompactionTask | None:
        if not connection.in_transaction():
            raise MemoryError("invalid_input")
        # Only committed nodes enter the queue; a caller rollback remains retryable.
        async with self._read() as reader:
            return await self._ready.next(reader, cutoff, self._view_size)

    async def complete_compaction(
        self, connection: AsyncConnection, task: CompactionTask, text: str
    ) -> Node:
        return await tree.complete_compaction(connection, task, text, self._view_size)

    async def view(
        self, cursor: str | None = None, cutoff: int | None = None
    ) -> ViewPage:
        offset = 0
        if cursor is None:
            async with self._read() as connection:
                admitted = await self._cutoff(connection, cutoff)
                state = (
                    (await connection.execute(select(memory_state))).mappings().one()
                )
                through = min(admitted, state["view_through"])
                full = await tree.load_frontier(
                    connection,
                    tree.frontier(state["main_parts"]),
                    state["view_through"],
                )
                nodes = await tree.clip_nodes(connection, full, 0, through)
                nodes, _ = await tree.coarsen(
                    connection,
                    nodes,
                    through,
                    MAIN_UPPER_BYTES,
                    MAIN_UPPER_BYTES,
                    False,
                    view_size=self._view_size,
                )
                fitting: list[Node] = []
                for item in nodes:
                    if (
                        tree.rendered_size(
                            [*fitting, item], item.start + item.count, self._view_size
                        )
                        > MAIN_UPPER_BYTES
                    ):
                        break
                    fitting.append(item)
                through = fitting[-1].start + fitting[-1].count if fitting else 0
                snapshot = _Snapshot(tuple(fitting), through, admitted)
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
                or not 0 <= offset <= len(snapshot.nodes)
                or len(cursor) > CURSOR_CHARACTERS
            ):
                raise MemoryError("stale_view")
            if cutoff is not None and snapshot.admitted_through > cutoff:
                raise MemoryError("invalid_input")
        page_nodes: list[Node] = []
        for item in snapshot.nodes[offset:]:
            next_offset = offset + len(page_nodes) + 1
            candidate = ViewPage(
                notice=NOTICE,
                nodes=(*page_nodes, item),
                through=snapshot.through,
                admitted_through=snapshot.admitted_through,
                cursor=f"{identifier}:{next_offset}"
                if next_offset < len(snapshot.nodes)
                else None,
            )
            if (
                max(
                    encoded_size(candidate.model_dump(mode="json")),
                    self._result_size(candidate.model_dump(mode="json")),
                )
                > RESULT_BYTES
            ):
                break
            page_nodes.append(item)
        if offset < len(snapshot.nodes) and not page_nodes:
            raise MemoryError("unavailable")
        next_offset = offset + len(page_nodes)
        return ViewPage(
            notice=NOTICE,
            nodes=tuple(page_nodes),
            through=snapshot.through,
            admitted_through=snapshot.admitted_through,
            cursor=f"{identifier}:{next_offset}"
            if next_offset < len(snapshot.nodes)
            else None,
        )

    async def zoom(
        self, start: int, count: int, cutoff: int | None = None
    ) -> ZoomResult:
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
                    notice=NOTICE,
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

    async def date(self, position: int, cutoff: int | None = None) -> DateResult:
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
                notice=NOTICE,
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
    ) -> None:
        await tree.rebuild(connection, start, count, self._view_size)
        self._ready = tree.Ready()
        self._snapshots.clear()

    async def embedding_candidates(self, limit: int = 50) -> tuple[Record, ...]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise MemoryError("invalid_input")
        async with self._read() as connection:
            cutoff = await self._cutoff(connection, None)
            selections: list[tuple[datetime, RecordReference]] = []
            for store, table in RECORD_TABLES.items():
                created = (
                    table.c.received_at
                    if store == "source_record"
                    else table.c.created_at
                )
                condition = and_(
                    retrieval.visible(store, cutoff),
                    table.c.embedding.is_(None),
                )
                if store == "source_record":
                    condition = and_(condition, conversational_text)
                rows = (
                    await connection.execute(
                        select(table.c.id, created)
                        .where(condition)
                        .order_by(created, table.c.id)
                        .limit(limit)
                    )
                ).all()
                selections.extend(
                    (
                        row[1],
                        RecordReference(store=store, id=row[0]),
                    )
                    for row in rows
                )
            selections.sort(key=lambda item: (item[0], item[1].store, item[1].id))
            result: list[Record] = []
            for _, identity in selections[:limit]:
                original = await retrieval.record(connection, identity, cutoff)
                if original is not None:
                    result.append(original)
            return tuple(result)

    async def update_embedding(
        self,
        connection: AsyncConnection,
        identity: RecordReference,
        embedding: Sequence[float],
    ) -> None:
        if not connection.in_transaction():
            raise MemoryError("invalid_input")
        value = validate_embedding(embedding)
        table = RECORD_TABLES[identity.store]
        condition = table.c.id == identity.id
        if identity.store == "source_record":
            condition = and_(condition, conversational_text)
        identifier = await connection.scalar(
            update(table).where(condition).values(embedding=value).returning(table.c.id)
        )
        if identifier is None:
            raise MemoryError("not_found")

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
                            source_conversation.c.capture_after_event_id,
                            source_conversation.c.checkpoint_event_id,
                            source_conversation.c.checkpoint_native_digest,
                            source_conversation.c.last_read_at,
                            source_conversation.c.capture_error,
                        ).order_by(
                            source_conversation.c.provider,
                            source_conversation.c.native_id,
                        )
                    )
                )
                .mappings()
                .all()
            )
            embeddings = 0
            for table in RECORD_TABLES.values():
                condition = table.c.embedding.is_(None)
                if table is source_record:
                    condition = and_(condition, conversational_text)
                embeddings += int(
                    await connection.scalar(
                        select(func.count()).select_from(table).where(condition)
                    )
                    or 0
                )

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
                "embedding_backlog": embeddings,
                "gaps": gaps,
                "lanes": [json_row(dict(row)) for row in lanes],
                "conversations": [json_row(dict(row)) for row in conversations],
            }
