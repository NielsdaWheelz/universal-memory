from __future__ import annotations

import os
import unittest
from uuid import uuid4

from sqlalchemy import insert, select, text
from sqlalchemy.ext.asyncio import create_async_engine

from universal_memory import (
    DreamNote,
    DreamResult,
    MemoryError,
    MemoryStore,
    OpenInput,
    SearchInput,
    Submission,
)
from universal_memory.schema import install_guards, memory_state, metadata


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
                with self.assertRaises(MemoryError) as failure:
                    await self.store.complete_compaction(connection, task, "\x00")
                self.assertEqual(failure.exception.code, "invalid_input")
                await self.store.complete_compaction(
                    connection, task, task.free_text or "attributed summary"
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
