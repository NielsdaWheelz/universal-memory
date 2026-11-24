from __future__ import annotations

import os
import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import insert, select, text
from sqlalchemy.ext.asyncio import create_async_engine

from universal_memory import (
    Conversation,
    DreamNote,
    DreamResult,
    Event,
    Lane,
    MemoryError,
    MemoryStore,
    Node,
    OpenInput,
    Record,
    SearchInput,
    Submission,
    normalize_event,
)
from universal_memory.policy import (
    ARCHIVE_GENERATION,
    NODE_HARD_BYTES,
    RESULT_BYTES,
    TOOL_OMISSION_MARKER,
    encoded_size,
)
from universal_memory.schema import install_guards, memory_state, metadata
from universal_memory.types import Occurrence


@unittest.skipUnless(
    os.getenv("MEMORY_TEST_DATABASE_URL"), "isolated postgres required"
)
class Completion(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(os.environ["MEMORY_TEST_DATABASE_URL"])
        async with self.engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.run_sync(metadata.drop_all)
            await connection.run_sync(metadata.create_all)
            await connection.execute(insert(memory_state).values(id=1))
            await install_guards(connection)
        self.store = MemoryStore(self.engine)

    async def asyncTearDown(self) -> None:
        await self.engine.dispose()

    async def test_note_tree_dream_and_repair_completion(self) -> None:
        submission = Submission(
            kind="native",
            machine="devbox",
            account="jarvis",
            provider="jarvis",
            submission_id=uuid4(),
        )
        async with self.engine.begin() as connection:
            receipt = await self.store.append_note(
                connection, "working hypothesis: the pilot failed", submission
            )
            self.assertEqual(
                receipt,
                await self.store.append_note(
                    connection, "working hypothesis: the pilot failed", submission
                ),
            )
            with self.assertRaises(MemoryError) as failure:
                await self.store.append_note(connection, "changed", submission)
            self.assertEqual(failure.exception.code, "submission_conflict")
        while True:
            async with self.engine.begin() as connection:
                task = await self.store.next_compaction(connection)
            if task is None:
                break
            async with self.engine.begin() as connection:
                changed = task.model_copy(
                    update={
                        "copies": (
                            task.copies[0].model_copy(update={"text": "\x00"}),
                            *task.copies[1:],
                        )
                    }
                )
                with self.assertRaises(MemoryError) as failure:
                    await self.store.complete_compaction(connection, changed, ())
                self.assertEqual(failure.exception.code, "source_conflict")
                await self.store.complete_compaction(
                    connection,
                    task,
                    tuple(
                        Node(
                            start=item.start,
                            count=item.count,
                            text="attributed summary",
                        )
                        for item in task.tasks
                    ),
                )
        view = await self.store.main_view(1)
        self.assertEqual((view[0].start, view[0].count), (0, 1))
        seed = await self.store.dream_input(cutoff=1)
        result = DreamResult(
            notes=(
                DreamNote(
                    text=(
                        "agent interpretation: a failed pilot "
                        "leaves the hypothesis unresolved"
                    ),
                    references=seed.references[:1],
                ),
            )
        )
        async with self.engine.begin() as connection:
            notes = await self.store.complete_dream(
                connection, seed, result, seed.references
            )
            self.assertEqual(len(notes), 1)
            self.assertEqual(
                await connection.scalar(select(memory_state.c.dream_through)), 1
            )
            self.assertEqual(
                await connection.scalar(select(memory_state.c.leaf_count)), 2
            )
        async with self.engine.begin() as connection:
            with self.assertRaises(MemoryError):
                await self.store.complete_dream(
                    connection, seed, result, seed.references
                )
            await self.store.rebuild(connection)
            self.assertEqual(
                await connection.scalar(select(memory_state.c.dream_through)), 1
            )
        vector = [1.0] + [0.0] * 1535
        found = await self.store.search(SearchInput(query="pilot"), vector)
        self.assertGreater(found.candidates, 0)
        opened = await self.store.open(
            OpenInput(identities=(found.records[0].reference,))
        )
        self.assertTrue(opened.records)

    async def test_orientation_recovers_retained_originals_and_late_history(
        self,
    ) -> None:
        now = datetime(2026, 11, 17, tzinfo=UTC)
        lane = Lane(machine="devbox", account="codex-work", provider="codex")
        context = "context " * (NODE_HARD_BYTES // len("context ") + 1)
        correction = Event(
            native_event_id="correction",
            native_digest="correction-digest",
            role="owner",
            kind="text",
            attributes={"secret_omissions": 0},
            text="correction: use cobalt for the project. " + context,
            occurred_at=now,
        )
        raw_tool_text = (
            "retained beginning "
            + "甲乙丙丁 " * 5_000
            + "unretained-middle-sentinel"
            + "甲乙丙丁 " * 5_000
            + " retained ending"
        )
        tool = normalize_event(
            {
                "native_event_id": "tool",
                "native_digest": "complete-native-tool-digest",
                "role": "tool",
                "kind": "tool_result",
                "attributes": {"native_call_id": "call", "outcome": "ok"},
                "text": raw_tool_text,
                "occurred_at": now,
            }
        )
        old_decision = correction.model_copy(
            update={
                "native_event_id": "old-decision",
                "native_digest": "old-decision-digest",
                "text": "use amber for the project. " + context,
                "occurred_at": now - timedelta(days=7),
            }
        )
        async with self.engine.begin() as connection:
            await self.store.activate_lane(connection, lane, (), now)
            main_lane = Lane(machine="devbox", account="jarvis", provider="jarvis")
            await self.store.activate_lane(connection, main_lane, (), now)
            (current,) = await self.store.sync_conversations(
                connection, main_lane, (Conversation(native_id="current"),), now, True
            )
            (late,) = await self.store.sync_conversations(
                connection,
                lane,
                (Conversation(native_id="late"),),
                now,
                True,
            )
            await self.store.append_main_events(
                connection, current.id, (correction, tool)
            )
            await self.store.append_occurrences(
                connection,
                lane,
                late.id,
                None,
                None,
                None,
                (
                    Occurrence(
                        native_event_id=old_decision.native_event_id,
                        native_digest=old_decision.native_digest,
                        capture_digest="f" * 64,
                        records=(old_decision,),
                    ),
                ),
                True,
                now + timedelta(hours=1),
            )
        while True:
            async with self.engine.begin() as connection:
                batch = await self.store.next_compaction(connection)
            if batch is None:
                break
            async with self.engine.begin() as connection:
                await self.store.complete_compaction(
                    connection,
                    batch,
                    tuple(
                        Node(
                            start=task.start,
                            count=task.count,
                            text="project discussion and output",
                        )
                        for task in batch.tasks
                    ),
                )

        view = await self.store.view()
        self.assertIsNone(view.cursor)
        self.assertEqual(view.through, view.admitted_through)
        self.assertNotIn("cobalt", " ".join(node.text for node in view.nodes))
        # The supplied view routes to a source part, not a complete native event.
        node = next(node for node in view.nodes if node.start == 1)
        zoomed = await self.store.zoom(
            node.start, node.count, generation=node.reference.generation
        )
        assert zoomed.record is not None and zoomed.full_event is not None
        self.assertEqual(zoomed.record.native_event_id, "tool")
        self.assertNotEqual(zoomed.record.text, tool.text)
        records: list[Record] = []
        cursor = None
        pages = 0
        while True:
            page = await self.store.open(
                OpenInput(event=zoomed.full_event, cursor=cursor)
            )
            pages += 1
            self.assertLessEqual(
                encoded_size(page.model_dump(mode="json")), RESULT_BYTES
            )
            for record in page.records:
                assert record.kind != "missing"
                records.append(record)
            part_count = records[0].part_count
            assert part_count is not None
            self.assertEqual(page.remaining, part_count - len(records))
            cursor = page.cursor
            if cursor is None:
                break
        self.assertGreater(pages, 1)
        self.assertEqual([record.part for record in records], list(range(len(records))))
        self.assertEqual("".join(record.text for record in records), tool.text)
        self.assertIn(TOOL_OMISSION_MARKER, tool.text)
        self.assertNotIn("unretained-middle-sentinel", tool.text)
        self.assertTrue(tool.text.startswith("retained beginning"))
        self.assertTrue(tool.text.endswith("retained ending"))
        for record in records:
            assert record.kind == "tool_result"
            self.assertFalse(record.clipped)
            self.assertEqual(record.native_digest, tool.native_digest)
            self.assertEqual(record.provenance.conversation_id, current.id)
            self.assertEqual(
                record.attributes.omitted_characters,
                len(raw_tool_text) - len(tool.text) + len(TOOL_OMISSION_MARKER),
            )

        found = await self.store.search(
            SearchInput(query="cobalt"), [1.0] + [0.0] * 1535
        )
        self.assertGreater(found.lexical_candidates, 0)
        opened = await self.store.open(
            OpenInput(identities=(found.records[0].reference,))
        )
        original = opened.records[0]
        assert original.kind == "text"
        self.assertEqual(original.role, "owner")
        self.assertEqual(original.text, correction.text)
        self.assertEqual(original.provenance.occurred_at, now)
        late_date = await self.store.date(
            view.through - 1, generation=ARCHIVE_GENERATION
        )
        self.assertEqual(late_date.occurred_at, now - timedelta(days=7))
        assert original.provenance.received_at is not None
        self.assertGreaterEqual(late_date.received_at, original.provenance.received_at)
        late_zoom = await self.store.zoom(
            view.through - 1, 1, generation=ARCHIVE_GENERATION
        )
        assert late_zoom.record is not None
        self.assertEqual(late_zoom.record.text, old_decision.text)
        self.assertEqual(late_date.received_at, late_zoom.record.provenance.received_at)
