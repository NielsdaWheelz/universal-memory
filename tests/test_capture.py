from __future__ import annotations

import json
import os
import unittest
from datetime import UTC, datetime, timedelta, timezone

from sqlalchemy import func, insert, select, text
from sqlalchemy.ext.asyncio import create_async_engine

from universal_memory import (
    Conversation,
    Event,
    EventIdentity,
    Lane,
    MemoryError,
    MemoryStore,
    OpenInput,
)
from universal_memory.policy import RESULT_BYTES, canonical_json
from universal_memory.schema import (
    install_guards,
    memory_leaf,
    memory_state,
    metadata,
    source_record,
)


def text_result_size(value: object) -> int:
    return len(json.dumps(canonical_json(value), ensure_ascii=False).encode())


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
            await self.store.append_events(
                connection, original.id, None, None, (event,), True, self.now
            )
            await transaction.rollback()
        async with self.engine.begin() as connection:
            self.assertEqual(
                await connection.scalar(select(memory_state.c.leaf_count)), 0
            )
            receipt = await self.store.append_events(
                connection, original.id, None, None, (event,), True, self.now
            )
            self.assertEqual(
                await connection.scalar(select(memory_state.c.leaf_count)), 2
            )
        async with self.engine.begin() as connection:
            with self.assertRaises(MemoryError) as failure:
                await self.store.append_events(
                    connection, original.id, None, None, (event,), True, self.now
                )
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
            copied = await self.store.append_events(
                connection, fork.id, None, None, (copy,), True, self.now
            )
            self.assertEqual(copied.checkpoint_event_id, "copy")
            self.assertEqual(copied.checkpoint_native_digest, "fork-native")
            self.assertEqual(
                await connection.scalar(select(memory_state.c.leaf_count)), 2
            )
            await self.store.append_events(
                connection, fork.id, "copy", "fork-native", (), True, self.now
            )
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
            role="tool",
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
                await self.store.append_events(
                    connection,
                    conversation.id,
                    None,
                    None,
                    (prefix, oversized, later),
                    True,
                    self.now,
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
            receipt = await self.store.append_events(
                connection, conversation.id, None, None, (prefix,), False, self.now
            )
        async with self.engine.begin() as connection:
            with self.assertRaises(MemoryError) as failure:
                await self.store.append_events(
                    connection,
                    conversation.id,
                    receipt.checkpoint_event_id,
                    receipt.checkpoint_native_digest,
                    (oversized,),
                    False,
                    self.now,
                )
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
            assert task is not None and task.free_text is not None
            await self.store.complete_compaction(connection, task, task.free_text)
        zoomed = await self.store.zoom(0, 1)
        assert zoomed.record is not None
        self.assertEqual(zoomed.record.text, prefix.text)
        self.assertLessEqual(
            text_result_size(zoomed.model_dump(mode="json")), RESULT_BYTES
        )
