from __future__ import annotations

import hashlib
import json
import os
import unittest
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

from llm_tools.schema import strict_encode
from sqlalchemy import func, insert, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from universal_memory import (
    Conversation,
    Event,
    EventIdentity,
    Lane,
    MemoryError,
    MemoryStore,
    OpenInput,
    collect_ready_view,
    memory_reference_attributes,
    render_ready_view,
)
from universal_memory.policy import (
    ARCHIVE_GENERATION,
    NOTICE,
    POSITION_NOTICE,
    RESULT_BYTES,
    canonical_json,
)
from universal_memory.schema import (
    install_guards,
    memory_leaf,
    memory_retired_source_record,
    memory_state,
    metadata,
    source_record,
)
from universal_memory.tools import (
    MEMORY_OPEN_SPEC,
    MEMORY_SEARCH_SPEC,
    MEMORY_ZOOM_SPEC,
)
from universal_memory.types import (
    ConversationReceipt,
    Missing,
    Occurrence,
    OpenPage,
    Provenance,
    RecordReference,
    SearchResult,
    Synthesis,
    ViewPage,
    ZoomResult,
    build_record,
)


def text_result_size(value: object) -> int:
    return len(json.dumps(canonical_json(value), ensure_ascii=False).encode())


class PublicReadResults(unittest.TestCase):
    def test_nonempty_records_fit_the_closed_tool_schemas(self) -> None:
        legacy = {"kind": "range", "start": 0, "count": 1}
        current = {**legacy, "generation": 2}
        refs = [legacy, current]
        stored_synthesis = {"seed_start": 0, "seed_end": 1, "references": refs}
        synthesis = Synthesis.model_validate_json(json.dumps(stored_synthesis))
        call = {"native_call_id": "call", "tool_name": "kept", "secret_omissions": 0}
        result = {
            "native_call_id": "call",
            "outcome": "ok",
            "omitted_characters": 0,
            "secret_omissions": 0,
        }
        cases = (
            ("text", {"secret_omissions": 0}),
            ("tool_call", call),
            ("tool_result", result),
            (
                "memory_reference",
                {
                    "tool_name": "memory.view",
                    "references": refs,
                    "omitted_reference_count": 0,
                },
            ),
            (
                "attachment",
                {
                    "secret_omissions": 0,
                    "reference": "a",
                    "media_type": "text/plain",
                    "text_availability": "full",
                },
            ),
            (
                "context",
                {
                    "secret_omissions": 0,
                    "context_kind": "other",
                    "body_retention": "reference_only",
                },
            ),
            ("gap", {"reason": "unreadable"}),
            ("revision", {"reason": "branch"}),
            ("note", None),
            ("summary", None),
        )
        records = tuple(
            build_record(
                store={"note": "memory_log", "summary": "memory_summary"}.get(
                    kind, "source_record"
                ),
                id=uuid4(),
                kind=kind,
                text="",
                role="unknown",
                provenance=Provenance(synthesis=synthesis if kind == "note" else None),
                attributes=attributes,
            )
            for kind, attributes in cases
        )
        search = SearchResult(
            notice=NOTICE,
            records=records,
            lexical_candidates=len(records),
            semantic_candidates=0,
            candidates=len(records),
            returned=len(records),
            omitted_limit=0,
            omitted_bytes=0,
            missing_embeddings=0,
        )
        opened = OpenPage(notice=NOTICE, records=records, cursor=None, remaining=0)
        for spec, page in ((MEMORY_SEARCH_SPEC, search), (MEMORY_OPEN_SPEC, opened)):
            self.assertEqual(
                strict_encode(spec.success_type, spec.success_schema, page),
                page.model_dump(mode="json"),
            )
        for record in records:
            zoom = ZoomResult(notice=POSITION_NOTICE, record=record)
            self.assertEqual(
                strict_encode(
                    MEMORY_ZOOM_SPEC.success_type, MEMORY_ZOOM_SPEC.success_schema, zoom
                ),
                zoom.model_dump(mode="json"),
            )
        output = opened.model_dump(mode="json")["records"]
        self.assertIsNone(output[0]["attributes"]["delivery"])
        self.assertIsNone(output[1]["attributes"]["omitted_characters"])
        self.assertIsNone(output[2]["attributes"]["source_output"])
        self.assertIsNone(output[2]["attributes"]["exit_code"])
        self.assertEqual(
            output[3]["attributes"]["references"],
            [{**legacy, "generation": 1}, current],
        )
        self.assertEqual(
            output[-2]["provenance"]["synthesis"],
            {
                **stored_synthesis,
                "generation": 1,
                "references": [{**legacy, "generation": 1}, current],
            },
        )
        self.assertEqual(synthesis.model_dump(mode="json"), stored_synthesis)


@unittest.skipUnless(
    os.getenv("MEMORY_TEST_DATABASE_URL"), "isolated postgres required"
)
class Capture(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(os.environ["MEMORY_TEST_DATABASE_URL"])
        async with self.engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.run_sync(metadata.drop_all)
            await connection.run_sync(metadata.create_all)
            await connection.execute(insert(memory_state).values(id=1))
            await install_guards(connection)
        self.store = MemoryStore(self.engine)
        self.lane = Lane(machine="devbox", account="codex-work", provider="codex")
        self.now = datetime.now(UTC)

    async def asyncTearDown(self) -> None:
        await self.engine.dispose()

    async def append(
        self,
        connection: AsyncConnection,
        receipt: ConversationReceipt,
        events: tuple[Event, ...],
        caught_up: bool,
    ) -> ConversationReceipt:
        return await self.store.append_occurrences(
            connection,
            self.lane,
            receipt.id,
            receipt.checkpoint_event_id,
            receipt.checkpoint_native_digest,
            receipt.checkpoint_capture_digest,
            tuple(
                Occurrence(
                    native_event_id=event.native_event_id,
                    native_digest=event.native_digest,
                    capture_digest=hashlib.sha256(
                        event.native_digest.encode()
                    ).hexdigest(),
                    records=(event,),
                )
                for event in events
            ),
            caught_up,
            self.now,
        )

    async def test_atomic_capture_retry_and_original_coverage(self) -> None:
        async with self.engine.begin() as connection:
            first = await self.store.activate_lane(
                connection,
                self.lane,
                (Conversation(native_id="original", head_event_id="old"),),
                self.now,
            )
            repeated = await self.store.activate_lane(
                connection,
                self.lane,
                (Conversation(native_id="original", head_event_id="later"),),
                self.now,
            )
            self.assertEqual(first, repeated)
            original, fork = await self.store.sync_conversations(
                connection,
                self.lane,
                (
                    Conversation(native_id="original"),
                    Conversation(
                        native_id="fork", relation="fork", parent_native_id="original"
                    ),
                ),
                self.now,
                True,
            )
        event = Event(
            native_event_id="new",
            native_digest="native-new",
            role="owner",
            kind="text",
            attributes={"secret_omissions": 0},
            text="a" * 8_001,
        )
        async with self.engine.connect() as connection:
            transaction = await connection.begin()
            await self.append(connection, original, (event,), True)
            await transaction.rollback()
        async with self.engine.begin() as connection:
            self.assertEqual(
                await connection.scalar(select(memory_state.c.leaf_count)), 0
            )
            receipt = await self.append(connection, original, (event,), True)
            self.assertEqual(
                await connection.scalar(select(memory_state.c.leaf_count)), 2
            )
        async with self.engine.begin() as connection:
            with self.assertRaises(MemoryError) as failure:
                await self.append(connection, original, (event,), True)
            self.assertEqual(failure.exception.code, "stale_checkpoint")
        copy = event.model_copy(
            update={
                "native_event_id": "copy",
                "native_digest": "fork-native",
                "attributes": {
                    "secret_omissions": 0,
                    "inherited_origin": {
                        "native_conversation_id": "original",
                        "native_event_id": "new",
                        "native_digest": "native-new",
                    },
                },
            }
        )
        async with self.engine.begin() as connection:
            copied = await self.append(connection, fork, (copy,), True)
            self.assertEqual(copied.checkpoint_event_id, "copy")
            self.assertEqual(copied.checkpoint_native_digest, "fork-native")
            self.assertEqual(
                await connection.scalar(select(memory_state.c.leaf_count)), 2
            )
            await self.append(connection, copied, (), True)
        self.assertEqual(receipt.capture_after_event_id, "old")

    async def test_complete_open_admission_preserves_prefix(self) -> None:
        self.store = MemoryStore(self.engine, result_size=text_result_size)
        async with self.engine.begin() as connection:
            await self.store.activate_lane(connection, self.lane, (), self.now)
            (conversation,) = await self.store.sync_conversations(
                connection,
                self.lane,
                (Conversation(native_id="admission"),),
                self.now,
                True,
            )
        prefix = Event(
            native_event_id="prefix",
            native_digest="prefix-digest",
            role="owner",
            kind="text",
            attributes={"secret_omissions": 0},
            text="retained preceding evidence",
            occurred_at=datetime(
                2026, 10, 9, 5, 30, tzinfo=timezone(timedelta(hours=5, minutes=30))
            ),
        )
        oversized = Event(
            native_event_id="oversized",
            native_digest="oversized-digest",
            role="owner",
            kind="attachment",
            attributes={
                "reference": "\x01" * 2_000,
                "media_type": "text/plain",
                "text_availability": "full",
                "secret_omissions": 0,
            },
            text="\x01" * 7_800,
        )
        later = prefix.model_copy(
            update={"native_event_id": "later", "native_digest": "later-digest"}
        )
        async with self.engine.begin() as connection:
            with self.assertRaises(MemoryError) as failure:
                await self.append(
                    connection, conversation, (prefix, oversized, later), True
                )
            self.assertEqual(failure.exception.code, "event_too_large")
            self.assertEqual(
                await connection.scalar(
                    select(func.count()).select_from(source_record)
                ),
                0,
            )
            self.assertEqual(
                await connection.scalar(select(func.count()).select_from(memory_leaf)),
                0,
            )
            self.assertEqual(
                await connection.scalar(select(memory_state.c.leaf_count)), 0
            )
        async with self.engine.begin() as connection:
            receipt = await self.append(connection, conversation, (prefix,), False)
        async with self.engine.begin() as connection:
            with self.assertRaises(MemoryError) as failure:
                await self.append(connection, receipt, (oversized,), False)
            self.assertEqual(failure.exception.code, "event_too_large")
            parked = await self.store.park_capture(
                connection, conversation.id, "prefix", "event_too_large"
            )
            self.assertEqual(parked.checkpoint_event_id, "prefix")
            self.assertEqual(parked.capture_error, "event_too_large")
            self.assertEqual(
                await connection.scalar(select(memory_state.c.leaf_count)), 1
            )
        opened = await self.store.open(
            OpenInput(
                event=EventIdentity(
                    conversation_id=conversation.id, native_event_id="prefix"
                )
            )
        )
        original = opened.records[0]
        assert original.kind != "missing"
        self.assertEqual(original.text, prefix.text)
        self.assertEqual(
            original.provenance.occurred_at, datetime(2026, 10, 9, tzinfo=UTC)
        )
        self.assertEqual(
            original.model_dump(mode="json")["provenance"]["occurred_at"],
            "2026-10-09T00:00:00Z",
        )
        self.assertLessEqual(
            text_result_size(opened.model_dump(mode="json")), RESULT_BYTES
        )
        async with self.engine.begin() as connection:
            task = await self.store.next_compaction(connection)
            assert task is not None and task.copies and not task.tasks
            await self.store.complete_compaction(connection, task, ())
        zoomed = await self.store.zoom(0, 1, generation=ARCHIVE_GENERATION)
        assert zoomed.record is not None
        self.assertEqual(zoomed.record.text, prefix.text)
        self.assertLessEqual(
            text_result_size(zoomed.model_dump(mode="json")), RESULT_BYTES
        )

    async def test_retired_addresses_never_reopen_another_original(self) -> None:
        live = Event(
            native_event_id="survivor",
            native_digest="live-digest",
            role="owner",
            kind="text",
            attributes={"secret_omissions": 0},
            text="hello",
        )
        retired = RecordReference(store="source_record", id=uuid4())
        unknown = RecordReference(store="source_record", id=uuid4())
        async with self.engine.begin() as connection:
            await self.store.activate_lane(connection, self.lane, (), self.now)
            (conversation,) = await self.store.sync_conversations(
                connection,
                self.lane,
                (Conversation(native_id="retirement"),),
                self.now,
                True,
            )
            receipt = await self.append(connection, conversation, (live,), True)
            await connection.execute(
                insert(memory_retired_source_record).values(
                    id=retired.id,
                    conversation_id=conversation.id,
                    native_event_id="old-tool",
                    part=0,
                )
            )
        async with self.engine.begin() as connection:
            fresh = live.model_copy(update={"native_event_id": "later"})
            retired_event = live.model_copy(update={"native_event_id": "old-tool"})
            with self.assertRaisesRegex(MemoryError, "^source_conflict$"):
                await self.append(connection, receipt, (fresh, retired_event), True)
            self.assertEqual(
                await connection.scalar(select(memory_state.c.leaf_count)), 1
            )
            with patch("universal_memory.archive.uuid4", return_value=retired.id):
                with self.assertRaisesRegex(MemoryError, "^source_conflict$"):
                    await self.append(connection, receipt, (fresh,), True)
            survivor_id = await connection.scalar(select(source_record.c.id))
            with patch("universal_memory.archive.uuid4", return_value=survivor_id):
                with self.assertRaises(IntegrityError):
                    await self.append(connection, receipt, (fresh,), True)
            # Retry the first member in the same transaction after the rejected batch.
            await self.append(connection, receipt, (fresh,), True)
        event = EventIdentity(
            conversation_id=conversation.id, native_event_id="old-tool"
        )
        with self.assertRaisesRegex(MemoryError, "^retired$"):
            await self.store.open(OpenInput(event=event))
        with self.assertRaisesRegex(MemoryError, "^not_found$"):
            await self.store.open(
                OpenInput(event=event.model_copy(update={"native_event_id": "unknown"}))
            )
        page = await self.store.open(OpenInput(identities=(retired, unknown)))
        self.assertEqual(
            [item.error for item in page.records if isinstance(item, Missing)],
            ["retired", "not_found"],
        )
        with self.assertRaisesRegex(MemoryError, "^retired_range$"):
            await self.store.date(0)
        with self.assertRaisesRegex(MemoryError, "^retired_range$"):
            await self.store.zoom(0, 1)
        self.assertEqual(
            (await self.store.date(0, generation=ARCHIVE_GENERATION)).position, 0
        )
        selector = {
            "event": None,
            "identities": [
                retired.model_dump(mode="json"),
                unknown.model_dump(mode="json"),
            ],
        }
        old_digest = hashlib.sha256(canonical_json(selector).encode()).hexdigest()
        with self.assertRaisesRegex(MemoryError, "^stale_view$"):
            await self.store.open(
                OpenInput(identities=(retired, unknown), cursor=f"{old_digest}:1:1")
            )

    async def test_positional_wires_preserve_their_original_generation(self) -> None:
        legacy = {"kind": "range", "start": 0, "count": 1}
        for notice, generation in ((NOTICE, 1), (POSITION_NOTICE, 2)):
            wire = {
                "notice": notice,
                "nodes": [{"start": 0, "count": 1, "text": "one"}],
                "through": 1,
                "admitted_through": 1,
                "recent": [],
                "cursor": None,
            }
            page = ViewPage.model_validate_json(json.dumps(wire))
            self.assertEqual(page.generation, generation)
            node = page.nodes[0]
            self.assertEqual(node.reference.generation, 1)
            self.assertEqual(node.reference_for(page.generation).generation, generation)
            expected = {**legacy, "generation": 2} if generation == 2 else legacy
            attrs = memory_reference_attributes("memory.view", {"value": wire})
            self.assertEqual(attrs["references"], [expected])

            async def no_more(cursor: str) -> ViewPage:
                raise AssertionError(cursor)

            view = await collect_ready_view(page, no_more)
            self.assertEqual(view.generation, generation)
            self.assertEqual("generation=2" in render_ready_view(view), generation == 2)
        self.assertLessEqual(len(POSITION_NOTICE.encode()), len(NOTICE.encode()))
        synthesis = {"seed_start": 0, "seed_end": 1, "references": [legacy]}
        self.assertEqual(
            Synthesis.model_validate_json(json.dumps(synthesis)).model_dump(
                mode="json"
            ),
            synthesis,
        )
