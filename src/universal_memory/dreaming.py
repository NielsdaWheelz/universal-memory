from __future__ import annotations

import json
from bisect import bisect_left
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from pydantic import JsonValue
from sqlalchemy import func, insert, literal, select, union_all, update
from sqlalchemy.ext.asyncio import AsyncConnection

from .archive import lock_state
from .policy import (
    DREAM_RESULT_BYTES,
    DREAM_SAMPLE_BYTES,
    DREAM_SEED_BYTES,
    MemoryError,
    canonical_json,
    encoded_size,
    validate_note,
)
from .retrieval import visible
from .schema import (
    RECORD_TABLES,
    conversational_text,
    memory_leaf,
    memory_log,
    memory_state,
    source_record,
)
from .tree import (
    ViewSize,
    clip_nodes,
    coarsen,
    frontier,
    key,
    load_frontier,
    node,
    render_dream_view,
    texts,
)
from .types import (
    DreamInput,
    DreamResult,
    Node,
    NoteReceipt,
    RecordReference,
    Reference,
    Sample,
)


async def _eligible(
    connection: AsyncConnection, nodes: Sequence[Node], syntheses: Sequence[int]
) -> list[Node]:
    result: list[Node] = []
    for item in nodes:
        index = bisect_left(syntheses, item.start)
        if index == len(syntheses) or syntheses[index] >= item.start + item.count:
            result.append(item)
        elif item.count > 1:
            half = item.count // 2
            keys = (key(item.start, half), key(item.start + half, half))
            children = await texts(connection, keys)
            if len(children) != 2:
                raise MemoryError("unavailable")
            result.extend(
                await _eligible(
                    connection,
                    [node(child, children[child]) for child in keys],
                    syntheses,
                )
            )
    return result


def _seed_size(nodes: Sequence[Node], start: int, end: int, view_size: ViewSize) -> int:
    return view_size(render_dream_view(nodes, start, end))


async def dream_input(
    connection: AsyncConnection, cutoff: int, view_size: ViewSize
) -> DreamInput:
    state = (await connection.execute(select(memory_state))).mappings().one()
    start = cast(int, state["dream_through"])
    if type(cutoff) is not int or not start <= cutoff <= state["leaf_count"]:
        raise MemoryError("invalid_input")
    if cutoff > state["view_through"]:
        raise MemoryError(
            "compaction_blocked" if state["compaction_error"] else "unavailable"
        )
    full = await load_frontier(
        connection, frontier(state["main_parts"]), state["view_through"]
    )
    interval = await clip_nodes(connection, full, start, cutoff)
    syntheses = (
        (
            await connection.execute(
                select(memory_leaf.c.position)
                .join(memory_log, memory_leaf.c.note_id == memory_log.c.id)
                .where(
                    memory_leaf.c.position >= start,
                    memory_leaf.c.position < cutoff,
                    memory_log.c.synthesis.is_not(None),
                )
                .order_by(memory_leaf.c.position)
            )
        )
        .scalars()
        .all()
    )
    eligible = await _eligible(connection, interval, syntheses)
    eligible, _ = await coarsen(
        connection,
        eligible,
        cutoff,
        DREAM_SEED_BYTES,
        DREAM_SEED_BYTES,
        True,
        lambda values: _seed_size(values, start, cutoff, view_size),
    )
    accepted: list[Node] = []
    end = cutoff
    for item in eligible:
        if _seed_size([*accepted, item], start, cutoff, view_size) > DREAM_SEED_BYTES:
            end = item.start
            break
        accepted.append(item)
    references: list[Reference] = [item.reference for item in accepted]
    # One ticket per conversational original, regardless of its part count.
    originals = (
        select(
            literal("source_record").label("store"),
            source_record.c.id.label("id"),
            memory_leaf.c.position.label("position"),
        )
        .join(memory_leaf, memory_leaf.c.source_record_id == source_record.c.id)
        .where(
            conversational_text,
            source_record.c.part == 0,
            memory_leaf.c.position + source_record.c.part_count <= start,
        )
    )
    notes = (
        select(
            literal("memory_log").label("store"),
            memory_log.c.id.label("id"),
            memory_leaf.c.position.label("position"),
        )
        .join(memory_leaf, memory_leaf.c.note_id == memory_log.c.id)
        .where(
            memory_log.c.synthesis.is_(None),
            memory_log.c.agent_submission.is_not(None),
            memory_leaf.c.position < start,
        )
    )
    candidates = union_all(originals, notes).subquery()
    selected = (
        (await connection.execute(select(candidates).order_by(func.random()).limit(4)))
        .mappings()
        .all()
    )
    sample: list[dict[str, JsonValue]] = []
    for choice in selected:
        position = cast(int, choice["position"])
        count = 1
        if choice["store"] == "source_record":
            count = int(
                await connection.scalar(
                    select(source_record.c.part_count).where(
                        source_record.c.id == choice["id"]
                    )
                )
                or 1
            )
        represented = await clip_nodes(connection, full, position, position + count)
        preview: list[Node] = []
        for item in represented:
            proposed: dict[str, JsonValue] = {
                "store": choice["store"],
                "id": str(choice["id"]),
                "nodes": [value.model_dump(mode="json") for value in [*preview, item]],
                "partial_coverage": len(preview) + 1 < len(represented),
            }
            if view_size(canonical_json([*sample, proposed])) > DREAM_SAMPLE_BYTES:
                break
            preview.append(item)
        if preview:
            sample.append(
                {
                    "store": choice["store"],
                    "id": str(choice["id"]),
                    "nodes": [item.model_dump(mode="json") for item in preview],
                    "partial_coverage": len(preview) != len(represented),
                }
            )
            references.append(RecordReference(store=choice["store"], id=choice["id"]))
            references.extend(item.reference for item in preview)
    return DreamInput(
        seed_start=start,
        seed_end=end,
        cutoff=cutoff,
        nodes=tuple(accepted),
        sample=tuple(Sample.model_validate_json(json.dumps(item)) for item in sample),
        references=tuple(dict.fromkeys(references)),
    )


async def complete_dream(
    connection: AsyncConnection,
    seed: DreamInput,
    result: DreamResult,
    allowed_references: Sequence[Reference],
) -> tuple[NoteReceipt, ...]:
    if encoded_size(result.model_dump(mode="json")) > DREAM_RESULT_BYTES or (
        not seed.nodes and result.notes
    ):
        raise MemoryError("invalid_input")
    available = set(seed.references) | set(allowed_references)
    for note in result.notes:
        validate_note(note.text)
        if len(set(note.references)) != len(note.references):
            raise MemoryError("invalid_input")
        for reference in note.references:
            if reference not in available:
                raise MemoryError("invalid_input")
            if (
                reference.kind == "range"
                and reference.start + reference.count > seed.cutoff
            ):
                raise MemoryError("invalid_input")
    state = await lock_state(connection)
    if state["dream_through"] != seed.seed_start:
        raise MemoryError("stale_checkpoint")
    if not seed.seed_start <= seed.seed_end <= seed.cutoff <= state["leaf_count"]:
        raise MemoryError("invalid_input")
    for reference in dict.fromkeys(
        item for note in result.notes for item in note.references
    ):
        if reference.kind == "record":
            table = RECORD_TABLES[reference.store]
            existing = await connection.scalar(
                select(table.c.id).where(
                    table.c.id == reference.id, visible(reference.store, seed.cutoff)
                )
            )
            if existing is None:
                raise MemoryError("invalid_input")
    receipts: list[NoteReceipt] = []
    position = cast(int, state["leaf_count"])
    for note in result.notes:
        identifier = uuid4()
        created_at = await connection.scalar(
            insert(memory_log)
            .values(
                id=identifier,
                text=note.text,
                synthesis={
                    "seed_start": seed.seed_start,
                    "seed_end": seed.seed_end,
                    "references": [
                        item.model_dump(mode="json") for item in note.references
                    ],
                },
            )
            .returning(memory_log.c.created_at)
        )
        await connection.execute(
            insert(memory_leaf).values(position=position, note_id=identifier)
        )
        position += 1
        receipts.append(
            NoteReceipt(
                id=identifier,
                created_at=cast(datetime, created_at),
                native_conversation_id=None,
            )
        )
    await connection.execute(
        update(memory_state)
        .where(memory_state.c.id == 1)
        .values(leaf_count=position, dream_through=seed.seed_end)
    )
    return tuple(receipts)


async def claim_nightly(
    connection: AsyncConnection,
    occurrence: datetime,
    attempted_at: datetime | None = None,
) -> bool:
    attempted = attempted_at or datetime.now(UTC)
    if (
        occurrence.utcoffset() is None
        or attempted.utcoffset() is None
        or attempted < occurrence
    ):
        raise MemoryError("invalid_input")
    state = await lock_state(connection)
    previous = state["last_nightly_attempt_at"]
    if previous is not None and previous >= occurrence:
        return False
    await connection.execute(
        update(memory_state)
        .where(memory_state.c.id == 1)
        .values(last_nightly_attempt_at=attempted)
    )
    return True
