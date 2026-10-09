from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from sqlalchemy import func, insert, select, update
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncConnection

from . import retrieval
from .policy import (
    BATCH_EVENTS,
    BODY_BYTES,
    OPEN_IDENTITIES,
    RESULT_BYTES,
    MemoryError,
    canonical_json,
    encoded_size,
    split_text,
    validate_association,
    validate_note,
)
from .schema import (
    memory_lane,
    memory_leaf,
    memory_log,
    memory_state,
    source_conversation,
    source_record,
)
from .types import (
    Conversation,
    ConversationReceipt,
    Event,
    Lane,
    LaneReceipt,
    NoteProducer,
    NoteReceipt,
    Provenance,
    Submission,
    build_record,
)


async def lock_state(connection: AsyncConnection) -> RowMapping:
    if not connection.in_transaction():
        raise MemoryError("invalid_input")
    state = (
        (await connection.execute(select(memory_state).with_for_update()))
        .mappings()
        .one()
    )
    return state


def conversation_receipt(
    row: RowMapping, lane: Lane | None = None
) -> ConversationReceipt:
    names = ConversationReceipt.model_fields.keys() - {"foreign"}
    return ConversationReceipt.model_validate(
        {
            **{key: row[key] for key in names},
            "foreign": lane is not None
            and (row["machine"], row["account"]) != (lane.machine, lane.account),
        }
    )


async def activate_lane(
    connection: AsyncConnection,
    lane: Lane,
    inventory: Sequence[Conversation],
    observed_at: datetime,
) -> LaneReceipt:
    if observed_at.utcoffset() is None:
        raise MemoryError("invalid_input")
    if len({item.native_id for item in inventory}) != len(inventory):
        raise MemoryError("invalid_input")
    if encoded_size([item.model_dump(mode="json") for item in inventory]) > BODY_BYTES:
        raise MemoryError("event_too_large")
    await lock_state(connection)
    row = (
        (
            await connection.execute(
                select(memory_lane).where(
                    memory_lane.c.machine == lane.machine,
                    memory_lane.c.account == lane.account,
                )
            )
        )
        .mappings()
        .one_or_none()
    )
    if row is not None:
        if row["provider"] != lane.provider:
            raise MemoryError("source_conflict")
        return LaneReceipt(
            lane=lane,
            activated_at=row["activated_at"],
            last_inventory_at=row["last_inventory_at"],
        )
    now = datetime.now(UTC)
    await connection.execute(
        insert(memory_lane).values(
            **lane.model_dump(), activated_at=now, last_inventory_at=now
        )
    )
    for item in inventory:
        await _discover(connection, lane, item, observed_at, baseline=True)
    return LaneReceipt(lane=lane, activated_at=now, last_inventory_at=now)


async def _discover(
    connection: AsyncConnection,
    lane: Lane,
    item: Conversation,
    observed_at: datetime,
    *,
    baseline: bool,
) -> ConversationReceipt:
    row = (
        (
            await connection.execute(
                select(source_conversation).where(
                    source_conversation.c.provider == lane.provider,
                    source_conversation.c.native_id == item.native_id,
                )
            )
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        row = (
            (
                await connection.execute(
                    insert(source_conversation)
                    .values(
                        id=uuid4(),
                        provider=lane.provider,
                        native_id=item.native_id,
                        machine=lane.machine,
                        account=lane.account,
                        relation=item.relation,
                        parent_native_id=item.parent_native_id,
                        working_directory=item.working_directory,
                        captured_from=observed_at,
                        capture_after_event_id=item.head_event_id if baseline else None,
                    )
                    .returning(source_conversation)
                )
            )
            .mappings()
            .one()
        )
    return conversation_receipt(row, lane)


async def sync_conversations(
    connection: AsyncConnection,
    lane: Lane,
    conversations: Sequence[Conversation],
    observed_at: datetime,
    complete: bool,
) -> tuple[ConversationReceipt, ...]:
    await lock_state(connection)
    row = (
        (
            await connection.execute(
                select(memory_lane).where(
                    memory_lane.c.machine == lane.machine,
                    memory_lane.c.account == lane.account,
                    memory_lane.c.provider == lane.provider,
                )
            )
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise MemoryError("forbidden")
    if observed_at.utcoffset() is None:
        raise MemoryError("invalid_input")
    if len({item.native_id for item in conversations}) != len(conversations):
        raise MemoryError("invalid_input")
    receipts = tuple(
        [
            await _discover(connection, lane, item, observed_at, baseline=False)
            for item in conversations
        ]
    )
    if complete:
        await connection.execute(
            update(memory_lane)
            .where(
                memory_lane.c.machine == lane.machine,
                memory_lane.c.account == lane.account,
            )
            .values(last_inventory_at=datetime.now(UTC))
        )
    return receipts


async def _complete_original(
    connection: AsyncConnection,
    conversation_id: UUID,
    native_event_id: str,
    native_digest: str,
) -> bool:
    rows = (
        await connection.execute(
            select(
                source_record.c.part,
                source_record.c.part_count,
                source_record.c.native_digest,
            )
            .where(
                source_record.c.conversation_id == conversation_id,
                source_record.c.native_event_id == native_event_id,
            )
            .order_by(source_record.c.part)
        )
    ).all()
    if not rows:
        return False
    if any(row.native_digest != native_digest for row in rows):
        raise MemoryError("source_conflict")
    if any(
        row.part_count != len(rows) or row.part != part for part, row in enumerate(rows)
    ):
        raise MemoryError("source_conflict")
    return True


async def append_events(
    connection: AsyncConnection,
    conversation_id: UUID,
    expected_checkpoint: str | None,
    checkpoint_native_digest: str | None,
    events: Sequence[Event],
    caught_up: bool,
    observed_at: datetime,
    result_size: retrieval.ResultSize,
) -> ConversationReceipt:
    if observed_at.utcoffset() is None or len(events) > BATCH_EVENTS:
        raise MemoryError("invalid_input")
    if len({item.native_event_id for item in events}) != len(events):
        raise MemoryError("invalid_input")
    if (
        encoded_size(
            {
                "conversation_id": str(conversation_id),
                "expected_checkpoint": expected_checkpoint,
                "checkpoint_native_digest": checkpoint_native_digest,
                "events": [item.model_dump(mode="json") for item in events],
                "caught_up": caught_up,
                "observed_at": observed_at.isoformat(),
            }
        )
        > BODY_BYTES
    ):
        raise MemoryError("event_too_large")
    state = await lock_state(connection)
    row = (
        (
            await connection.execute(
                select(source_conversation)
                .where(source_conversation.c.id == conversation_id)
                .with_for_update()
            )
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise MemoryError("not_found")
    if row["checkpoint_event_id"] != expected_checkpoint:
        raise MemoryError("stale_checkpoint")
    if row["capture_error"] is not None:
        raise MemoryError(cast(str, row["capture_error"]))
    if row["checkpoint_native_digest"] != checkpoint_native_digest:
        raise MemoryError("source_conflict")
    retained: list[Event] = []
    checkpoint = expected_checkpoint
    digest = checkpoint_native_digest
    for event in events:
        if event.native_event_id == expected_checkpoint:
            raise MemoryError("invalid_input")
        if await _complete_original(
            connection, conversation_id, event.native_event_id, event.native_digest
        ):
            continue
        origin = event.attributes.get("inherited_origin")
        omit = False
        if isinstance(origin, dict) and row["relation"] == "fork":
            original_id = await connection.scalar(
                select(source_conversation.c.id).where(
                    source_conversation.c.provider == row["provider"],
                    source_conversation.c.native_id == origin["native_conversation_id"],
                )
            )
            if original_id is not None:
                # A conflicting origin proves no coverage for this fork.
                original_digest = await connection.scalar(
                    select(source_record.c.native_digest).where(
                        source_record.c.conversation_id == original_id,
                        source_record.c.native_event_id == origin["native_event_id"],
                        source_record.c.part == 0,
                    )
                )
                if original_digest == origin["native_digest"]:
                    omit = await _complete_original(
                        connection,
                        original_id,
                        cast(str, origin["native_event_id"]),
                        cast(str, origin["native_digest"]),
                    )
        if not omit:
            retained.append(event)
        checkpoint, digest = event.native_event_id, event.native_digest
    sequence = int(
        await connection.scalar(
            select(func.coalesce(func.max(source_record.c.sequence) + 1, 0)).where(
                source_record.c.conversation_id == conversation_id
            )
        )
        or 0
    )
    position = cast(int, state["leaf_count"])
    records: list[dict[str, object]] = []
    leaves: list[dict[str, object]] = []
    received_at = await connection.scalar(select(func.current_timestamp()))
    for event in retained:
        parts = split_text(event.text)
        total = max(OPEN_IDENTITIES, len(parts))
        next_offset = max(
            range(1, total),
            key=lambda offset: len(str(offset)) + len(str(total - offset)),
        )
        for part, text in enumerate(parts):
            record_id = uuid4()
            original = build_record(
                store="source_record",
                id=record_id,
                kind=event.kind,
                role=event.role,
                text=text,
                provenance=Provenance(
                    provider=row["provider"],
                    machine=row["machine"],
                    account=row["account"],
                    conversation_id=conversation_id,
                    native_conversation_id=row["native_id"],
                    occurred_at=event.occurred_at,
                    received_at=received_at,
                ),
                native_event_id=event.native_event_id,
                native_digest=event.native_digest,
                turn_id=event.turn_id,
                native_parent_id=event.native_parent_id,
                part=part,
                part_count=len(parts),
                attributes=event.attributes,
            )
            # The corpus cutoff can grow; selector size and event parts are bounded.
            page = retrieval.open_page(
                (original,),
                "f" * 64,
                2**63 - 1,
                next_offset,
                total - next_offset,
            )
            for result in (page, retrieval.leaf_zoom(original)):
                value = result.model_dump(mode="json")
                if max(encoded_size(value), result_size(value)) > RESULT_BYTES:
                    raise MemoryError("event_too_large")
            records.append(
                {
                    "id": record_id,
                    "conversation_id": conversation_id,
                    "sequence": sequence,
                    "native_event_id": event.native_event_id,
                    "part": part,
                    "part_count": len(parts),
                    "native_digest": event.native_digest,
                    "turn_id": event.turn_id,
                    "native_parent_id": event.native_parent_id,
                    "role": event.role,
                    "kind": event.kind,
                    "attributes": event.attributes,
                    "text": text,
                    "occurred_at": event.occurred_at,
                    "received_at": received_at,
                }
            )
            leaves.append({"position": position, "source_record_id": record_id})
            sequence += 1
            position += 1
    if records:
        await connection.execute(insert(source_record), records)
        await connection.execute(insert(memory_leaf), leaves)
    await connection.execute(
        update(memory_state).where(memory_state.c.id == 1).values(leaf_count=position)
    )
    values: dict[str, object] = {
        "checkpoint_event_id": checkpoint,
        "checkpoint_native_digest": digest,
    }
    if caught_up:
        values["last_read_at"] = datetime.now(UTC)
    updated = (
        (
            await connection.execute(
                update(source_conversation)
                .where(source_conversation.c.id == conversation_id)
                .values(**values)
                .returning(source_conversation)
            )
        )
        .mappings()
        .one()
    )
    return conversation_receipt(updated)


def note_id(submission: NoteProducer) -> UUID:
    identity = (
        [submission.machine, submission.account, str(submission.submission_id)]
        if isinstance(submission, Submission)
        else [
            submission.client,
            str(submission.owner_user_id),
            str(submission.submission_id),
        ]
    )
    return uuid5(
        NAMESPACE_URL,
        "urn:jarvis:memory-save-note:" + canonical_json(identity),
    )


async def recover_note(
    connection: AsyncConnection, text: str, submission: NoteProducer
) -> NoteReceipt | None:
    validate_note(text)
    validate_association(submission.native_conversation_id)
    row = (
        (
            await connection.execute(
                select(memory_log).where(memory_log.c.id == note_id(submission))
            )
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        return None
    if (
        row["text"] != text
        or row["agent_submission"] != submission.model_dump(mode="json")
        or row["synthesis"] is not None
    ):
        raise MemoryError("submission_conflict")
    return NoteReceipt(
        id=row["id"],
        created_at=row["created_at"],
        native_conversation_id=submission.native_conversation_id,
    )


async def append_note(
    connection: AsyncConnection,
    text: str,
    submission: NoteProducer,
) -> NoteReceipt:
    validate_note(text)
    validate_association(submission.native_conversation_id)
    state = await lock_state(connection)
    identifier = note_id(submission)
    provenance = submission.model_dump(mode="json")
    row = (
        (
            await connection.execute(
                select(memory_log).where(memory_log.c.id == identifier)
            )
        )
        .mappings()
        .one_or_none()
    )
    if row is not None:
        if (
            row["text"] != text
            or row["agent_submission"] != provenance
            or row["synthesis"] is not None
        ):
            raise MemoryError("submission_conflict")
    else:
        row = (
            (
                await connection.execute(
                    insert(memory_log)
                    .values(id=identifier, text=text, agent_submission=provenance)
                    .returning(memory_log)
                )
            )
            .mappings()
            .one()
        )
        position = cast(int, state["leaf_count"])
        await connection.execute(
            insert(memory_leaf).values(position=position, note_id=identifier)
        )
        await connection.execute(
            update(memory_state)
            .where(memory_state.c.id == 1)
            .values(leaf_count=position + 1)
        )
    return NoteReceipt(
        id=identifier,
        created_at=row["created_at"],
        native_conversation_id=submission.native_conversation_id,
    )


async def park_capture(
    connection: AsyncConnection,
    conversation_id: UUID,
    expected_checkpoint: str | None,
    code: str,
) -> ConversationReceipt:
    if code not in {
        "native_missing",
        "unsupported",
        "event_too_large",
        "history_changed",
        "activation_boundary_lost",
        "source_conflict",
    }:
        raise MemoryError("invalid_input")
    await lock_state(connection)
    row = (
        (
            await connection.execute(
                select(source_conversation)
                .where(source_conversation.c.id == conversation_id)
                .with_for_update()
            )
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise MemoryError("not_found")
    if row["checkpoint_event_id"] != expected_checkpoint:
        raise MemoryError("stale_checkpoint")
    updated = (
        (
            await connection.execute(
                update(source_conversation)
                .where(source_conversation.c.id == conversation_id)
                .values(capture_error=code)
                .returning(source_conversation)
            )
        )
        .mappings()
        .one()
    )
    return conversation_receipt(updated)


async def retry_capture(connection: AsyncConnection, conversation_id: UUID) -> None:
    await lock_state(connection)
    identifier = await connection.scalar(
        update(source_conversation)
        .where(source_conversation.c.id == conversation_id)
        .values(capture_error=None)
        .returning(source_conversation.c.id)
    )
    if identifier is None:
        raise MemoryError("not_found")
