from __future__ import annotations

import heapq
from collections.abc import Callable, Mapping, Sequence
from functools import cmp_to_key
from itertools import pairwise
from typing import cast

from llm_tools import prompt_json_text
from sqlalchemy import delete, insert, select, text, tuple_, update
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncConnection

from .archive import lock_state
from .policy import (
    COMPACT_LOWER_BYTES,
    COMPACT_UPPER_BYTES,
    COMPACTION_BATCH_SIZE,
    COMPACTION_READY_WINDOW,
    MAIN_LOWER_BYTES,
    MAIN_UPPER_BYTES,
    NODE_HARD_BYTES,
    MemoryError,
    bounded_text,
    canonical_json,
    utf8_size,
)
from .schema import (
    memory_leaf,
    memory_log,
    memory_node,
    memory_state,
    memory_summary,
    source_conversation,
    source_record,
)
from .types import CompactionBatch, CompactionTask, Node, RangeReference

type Key = tuple[int, int]
type ViewSize = Callable[[str], int]


def key(start: int, count: int) -> Key:
    RangeReference(start=start, count=count)
    return count.bit_length() - 1, start // count


def node(key: Key, text: str) -> Node:
    level, index = key
    return Node(start=index * 2**level, count=2**level, text=text)


def _render_lines(nodes: Sequence[Node]) -> str:
    return "\n".join(
        f"{item.start}+{item.count}|{prompt_json_text(item.text)}" for item in nodes
    )


def render(nodes: Sequence[Node], through: int) -> str:
    return f"historical evidence\n{_render_lines(nodes)}\ncoverage [0,{through})"


def render_dream_view(nodes: Sequence[Node], start: int, end: int) -> str:
    return (
        f"new historical evidence\n{_render_lines(nodes)}\n"
        f"eligible physical interval [{start},{end})"
    )


def rendered_size(
    nodes: Sequence[Node], through: int, view_size: ViewSize = utf8_size
) -> int:
    return view_size(render(nodes, through))


async def texts(connection: AsyncConnection, keys: Sequence[Key]) -> dict[Key, str]:
    if not keys:
        return {}
    rows = (
        (
            await connection.execute(
                select(memory_node).where(
                    tuple_(memory_node.c.level, memory_node.c.node_index).in_(keys)
                )
            )
        )
        .mappings()
        .all()
    )
    return {
        (cast(int, row["level"]), cast(int, row["node_index"])): cast(str, row["text"])
        for row in rows
    }


def frontier(value: object) -> list[Key]:
    if not isinstance(value, list):
        raise MemoryError("compaction_blocked")
    return [cast(Key, tuple(item)) for item in cast(list[list[int]], value)]


async def load_frontier(
    connection: AsyncConnection, parts: Sequence[Key], through: int
) -> tuple[Node, ...]:
    built = await texts(connection, parts)
    result: list[Node] = []
    position = 0
    for item in parts:
        if item not in built:
            raise MemoryError("compaction_blocked")
        value = node(item, built[item])
        if value.start != position:
            raise MemoryError("compaction_blocked")
        result.append(value)
        position += value.count
    if position != through:
        raise MemoryError("compaction_blocked")
    return tuple(result)


async def clip_nodes(
    connection: AsyncConnection, nodes: Sequence[Node], start: int, end: int
) -> list[Node]:
    result: list[Node] = []
    for item in nodes:
        right = item.start + item.count
        if right <= start or item.start >= end:
            continue
        if item.start >= start and right <= end:
            result.append(item)
        else:
            if item.count == 1:
                raise MemoryError("invalid_input")
            child_count = item.count // 2
            child_keys = (
                key(item.start, child_count),
                key(item.start + child_count, child_count),
            )
            children = await texts(connection, child_keys)
            if len(children) != 2:
                raise MemoryError("unavailable")
            result.extend(
                await clip_nodes(
                    connection,
                    [node(child, children[child]) for child in child_keys],
                    start,
                    end,
                )
            )
    return result


def _priority(left: tuple[int, int], right: tuple[int, int]) -> int:
    # Inclusive last-position age; exact cross-products avoid floating-point ties.
    difference = left[0] * right[1] - right[0] * left[1]
    return -1 if difference > 0 else 1 if difference < 0 else 0


async def coarsen(
    connection: AsyncConnection,
    nodes: list[Node],
    through: int,
    lower: int,
    upper: int,
    shrinking: bool,
    measure: Callable[[Sequence[Node]], int] | None = None,
    *,
    view_size: ViewSize = utf8_size,
) -> tuple[list[Node], bool]:
    size: Callable[[Sequence[Node]], int] = measure or (
        lambda values: rendered_size(values, through, view_size)
    )
    shrinking = shrinking or size(nodes) > upper
    while shrinking and size(nodes) > lower:
        pairs: list[tuple[int, Key, tuple[int, int]]] = []
        for index, (left, right) in enumerate(pairwise(nodes)):
            if (
                left.count == right.count
                and left.start % (2 * left.count) == 0
                and right.start == left.start + left.count
            ):
                pairs.append(
                    (
                        index,
                        key(left.start, left.count * 2),
                        (through - left.start - 2 * left.count + 1, left.count),
                    )
                )
        available = await texts(connection, [item[1] for item in pairs])
        choices = [item for item in pairs if item[1] in available]
        if not choices:
            break
        choices.sort(key=cmp_to_key(lambda a, b: _priority(a[2], b[2]) or a[0] - b[0]))
        index, parent, _ = choices[0]
        nodes[index : index + 2] = [node(parent, available[parent])]
    return nodes, shrinking and size(nodes) > lower


async def advance_views(
    connection: AsyncConnection, state: RowMapping, view_size: ViewSize
) -> None:
    through = cast(int, state["view_through"])
    main = list(await load_frontier(connection, frontier(state["main_parts"]), through))
    compact = list(
        await load_frontier(connection, frontier(state["compact_parts"]), through)
    )
    main_shrinking = cast(bool, state["main_shrinking"])
    compact_shrinking = cast(bool, state["compact_shrinking"])
    while through < cast(int, state["leaf_count"]):
        value = await connection.scalar(
            select(memory_node.c.text).where(
                memory_node.c.level == 0, memory_node.c.node_index == through
            )
        )
        if value is None:
            break
        main.append(Node(start=through, count=1, text=value))
        compact.append(Node(start=through, count=1, text=value))
        through += 1
        main, main_shrinking = await coarsen(
            connection,
            main,
            through,
            MAIN_LOWER_BYTES,
            MAIN_UPPER_BYTES,
            main_shrinking,
            view_size=view_size,
        )
        compact, compact_shrinking = await coarsen(
            connection,
            compact,
            through,
            COMPACT_LOWER_BYTES,
            COMPACT_UPPER_BYTES,
            compact_shrinking,
            view_size=view_size,
        )
    # Parent completion resumes unfinished shrink even with no new leaf.
    main, main_shrinking = await coarsen(
        connection,
        main,
        through,
        MAIN_LOWER_BYTES,
        MAIN_UPPER_BYTES,
        main_shrinking,
        view_size=view_size,
    )
    compact, compact_shrinking = await coarsen(
        connection,
        compact,
        through,
        COMPACT_LOWER_BYTES,
        COMPACT_UPPER_BYTES,
        compact_shrinking,
        view_size=view_size,
    )
    await connection.execute(
        update(memory_state)
        .where(memory_state.c.id == 1)
        .values(
            view_through=through,
            main_parts=[list(key(item.start, item.count)) for item in main],
            compact_parts=[list(key(item.start, item.count)) for item in compact],
            main_shrinking=main_shrinking,
            compact_shrinking=compact_shrinking,
        )
    )


async def source(connection: AsyncConnection, start: int, count: int) -> str:
    if count > 1:
        half = count // 2
        children = (key(start, half), key(start + half, half))
        values = await texts(connection, children)
        if len(values) != 2:
            raise MemoryError("unavailable")
        return "\n".join(
            f"{item.start}+{item.count}|{item.text}"
            for item in [node(child, values[child]) for child in children]
        )
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
    if leaf["note_id"] is not None:
        row = (
            (
                await connection.execute(
                    select(memory_log).where(memory_log.c.id == leaf["note_id"])
                )
            )
            .mappings()
            .one()
        )
        provenance = row["agent_submission"] or {"dreamer": row["synthesis"]}
        return (
            f"note {row['id']}; {canonical_json(provenance)}; "
            f"created {cast(object, row['created_at'])}\n{row['text']}"
        )
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
                .where(source_record.c.id == leaf["source_record_id"])
            )
        )
        .mappings()
        .one()
    )
    source = "/".join(
        str(value)
        for value in (row["provider"], row["machine"], row["account"])
        if value is not None
    )
    header = (
        f"source {row['id']}; {source}; "
        f"{row['role']}/{row['kind']}; conversation {row['native_id']}; "
        f"event {row['native_event_id']}; part {row['part']}/{row['part_count']}; "
        f"occurred {row['occurred_at']}; received {row['received_at']}"
    )
    attributes = cast(Mapping[str, object], row["attributes"])
    if isinstance(attributes.get("nexus"), dict):
        nexus = cast(dict[str, object], attributes["nexus"])
        attributes = {
            **attributes,
            "nexus": {
                name: value
                for name, value in nexus.items()
                if name
                not in {"capture_id", "export_seq", "envelope_sha256", "event_count"}
            },
        }
        if nexus["tool"] is not None:
            header += "; stored tool receipt"
    if (
        row["kind"] != "text"
        or attributes.get("inherited_origin") is not None
        or attributes.get("nexus") is not None
    ):
        header += "; " + canonical_json(attributes)
    return header + "\n" + cast(str, row["text"])


async def context_view(
    connection: AsyncConnection, cutoff: int, view_size: ViewSize
) -> tuple[Node, ...]:
    state = (await connection.execute(select(memory_state))).mappings().one()
    if type(cutoff) is not int or not 0 <= cutoff <= state["leaf_count"]:
        raise MemoryError("invalid_input")
    if cutoff > state["view_through"]:
        raise MemoryError(
            "compaction_blocked" if state["compaction_error"] else "unavailable"
        )
    full = await load_frontier(
        connection, frontier(state["main_parts"]), state["view_through"]
    )
    clipped = await clip_nodes(connection, full, 0, cutoff)
    clipped, _ = await coarsen(
        connection,
        clipped,
        cutoff,
        MAIN_LOWER_BYTES,
        MAIN_UPPER_BYTES,
        False,
        view_size=view_size,
    )
    if rendered_size(clipped, cutoff, view_size) > MAIN_UPPER_BYTES:
        raise MemoryError(
            "compaction_blocked" if state["compaction_error"] else "unavailable"
        )
    return tuple(clipped)


class Ready:
    """A startup reconstruction and bounded leaf window; database commits are truth."""

    def __init__(self) -> None:
        self.built: set[Key] | None = None
        self.queue: list[tuple[int, int, Key]] = []
        self.queued: set[Key] = set()
        self.leaf_cursor = 0

    def _push(self, item: Key) -> None:
        assert self.built is not None
        if item not in self.built and item not in self.queued:
            level, index = item
            heapq.heappush(self.queue, (index * 2**level, -(2**level), item))
            self.queued.add(item)

    def _parent(self, item: Key, total: int) -> None:
        assert self.built is not None
        level, index = item
        if (level, index ^ 1) in self.built and (index // 2 + 1) * 2 ** (
            level + 1
        ) <= total:
            self._push((level + 1, index // 2))

    def _fill(self, total: int) -> None:
        while (
            sum(item[0] == 0 for item in self.queued) < COMPACTION_READY_WINDOW
            and self.leaf_cursor < total
        ):
            self._push((0, self.leaf_cursor))
            self.leaf_cursor += 1

    async def next(
        self, connection: AsyncConnection, cutoff: int | None, view_size: ViewSize
    ) -> CompactionBatch | None:
        state = (await connection.execute(select(memory_state))).mappings().one()
        total = cast(int, state["leaf_count"])
        if cutoff is not None and not 0 <= cutoff <= total:
            raise MemoryError("invalid_input")
        if state["compaction_error"] is not None:
            raise MemoryError("compaction_blocked")
        if self.built is None:
            self.built = set(
                (
                    await connection.execute(
                        select(memory_node.c.level, memory_node.c.node_index)
                    )
                )
                .tuples()
                .all()
            )
            for item in self.built:
                self._parent(item, total)
        self._fill(total)
        selected: set[Key] = set()
        tasks: list[CompactionTask] = []
        copies: list[Node] = []
        while self.queue:
            eligible = [
                value
                for value in self.queue
                if value[2] not in selected
                and (cutoff is None or value[0] + -value[1] <= cutoff)
            ]
            if not eligible:
                break
            position, negative_count, item = min(eligible)
            existing = await connection.scalar(
                select(memory_node.c.text).where(
                    memory_node.c.level == item[0], memory_node.c.node_index == item[1]
                )
            )
            if existing is not None:
                self.queue.remove((position, negative_count, item))
                heapq.heapify(self.queue)
                self.queued.remove(item)
                self.built.add(item)
                self._parent(item, total)
                self._fill(total)
                continue
            count = -negative_count
            original = await source(connection, position, count)
            if len(original.encode()) <= NODE_HARD_BYTES:
                if len(copies) == COMPACTION_READY_WINDOW:
                    break
                copies.append(node(item, original))
            else:
                if len(tasks) == COMPACTION_BATCH_SIZE:
                    break
                tasks.append(
                    CompactionTask(start=position, count=count, source=original)
                )
            selected.add(item)
        if not tasks and not copies:
            return None
        context_through = min(
            state["view_through"], min((task.start for task in tasks), default=0)
        )
        compact = await load_frontier(
            connection, frontier(state["compact_parts"]), state["view_through"]
        )
        prefix = await clip_nodes(connection, compact, 0, context_through)
        fitting: list[Node] = []
        actual = 0
        for item in prefix:
            if (
                rendered_size([*fitting, item], item.start + item.count, view_size)
                > COMPACT_UPPER_BYTES
            ):
                break
            fitting.append(item)
            actual = item.start + item.count
        return CompactionBatch(
            tasks=tuple(tasks),
            copies=tuple(copies),
            context=render(fitting, actual),
            context_through=actual,
        )


async def complete_compaction(
    connection: AsyncConnection,
    batch: CompactionBatch,
    results: Sequence[Node],
    view_size: ViewSize,
) -> tuple[Node, ...]:
    state = await lock_state(connection)
    expected = {key(task.start, task.count): task for task in batch.tasks}
    returned = {key(result.start, result.count): result for result in results}
    if (
        (not batch.tasks and not batch.copies)
        or len(expected) != len(batch.tasks)
        or len(returned) != len(results)
        or expected.keys() != returned.keys()
    ):
        raise MemoryError("invalid_input")
    accepted: dict[Key, Node] = {}
    for copied in batch.copies:
        coordinates = key(copied.start, copied.count)
        if (
            coordinates in accepted
            or coordinates in expected
            or copied.start + copied.count > state["leaf_count"]
        ):
            raise MemoryError("invalid_input")
        original = await source(connection, copied.start, copied.count)
        if copied.text != original:
            raise MemoryError("source_conflict")
        bounded_text(copied.text, NODE_HARD_BYTES)
        accepted[coordinates] = copied
    for coordinates, task in expected.items():
        if task.start + task.count > state["leaf_count"]:
            raise MemoryError("invalid_input")
        original = await source(connection, task.start, task.count)
        if original != task.source or len(original.encode()) <= NODE_HARD_BYTES:
            raise MemoryError("source_conflict")
        result = returned[coordinates]
        accepted[coordinates] = Node(
            start=result.start,
            count=result.count,
            text=bounded_text(result.text.strip(), NODE_HARD_BYTES),
        )
    existing = await texts(connection, tuple(accepted))
    if any(existing[item] != accepted[item].text for item in existing):
        raise MemoryError("source_conflict")
    values = [
        {"level": item[0], "node_index": item[1], "text": value.text}
        for item, value in accepted.items()
        if item not in existing
    ]
    if values:
        await connection.execute(insert(memory_node), values)
    await advance_views(connection, state, view_size)
    return tuple(sorted(accepted.values(), key=lambda item: (item.start, -item.count)))


async def park_compaction(
    connection: AsyncConnection, task: CompactionTask, code: str
) -> None:
    if code not in {"invalid_completion", "candidate_too_large", "inference_failed"}:
        raise MemoryError("invalid_input")
    await lock_state(connection)
    await connection.execute(
        update(memory_state)
        .where(memory_state.c.id == 1)
        .values(
            compaction_error={"start": task.start, "count": task.count, "code": code}
        )
    )


async def retry_compression(connection: AsyncConnection) -> None:
    await lock_state(connection)
    await connection.execute(
        update(memory_state).where(memory_state.c.id == 1).values(compaction_error=None)
    )


async def rebuild(
    connection: AsyncConnection,
    start: int | None,
    count: int | None,
    view_size: ViewSize,
) -> None:
    state = await lock_state(connection)
    if (start is None) != (count is None):
        raise MemoryError("invalid_input")
    if start is None:
        await connection.execute(delete(memory_node))
        for table in (memory_log, memory_summary, source_record):
            await connection.execute(update(table).values(embedding=None))
            await connection.execute(text(f"REINDEX TABLE {table.name}"))
    else:
        assert count is not None
        coordinates = key(start, count)
        if start + count > state["leaf_count"]:
            raise MemoryError("invalid_input")
        ancestors: list[Key] = []
        level, index = coordinates
        while (index + 1) * 2**level <= state["leaf_count"]:
            ancestors.append((level, index))
            index //= 2
            level += 1
        await connection.execute(
            delete(memory_node).where(
                tuple_(memory_node.c.level, memory_node.c.node_index).in_(ancestors)
            )
        )
    await connection.execute(
        update(memory_state)
        .where(memory_state.c.id == 1)
        .values(
            view_through=0,
            main_parts=[],
            compact_parts=[],
            main_shrinking=False,
            compact_shrinking=False,
            compaction_error=None,
        )
    )
    refreshed = (await connection.execute(select(memory_state))).mappings().one()
    await advance_views(connection, refreshed, view_size)
