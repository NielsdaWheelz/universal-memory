"""Closed nexus exports and their atomic projection into the shared archive."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal, Self, cast
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, model_validator
from sqlalchemy import insert, select, update
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncConnection

from . import archive, retrieval
from .policy import BODY_BYTES, MemoryError, canonical_json, encoded_size
from .schema import memory_leaf, source_conversation, source_record
from .types import (
    Closed,
    Event,
    NexusAttributes,
    NexusDigest,
    NexusProgress,
    NexusSequence,
    NexusStatus,
    NexusTime,
    NexusUUID,
    normalize_event,
)

NEXUS_CONTRACT_REVISION = "nexus-turn-capture-v2"
type NexusRevision = Literal["nexus-turn-capture-v2"]


class NexusMessage(Closed):
    message_id: NexusUUID
    message_seq: NexusSequence
    created_at: NexusTime
    text: str


class NexusSettlement(Closed):
    status: NexusStatus
    completed_at: NexusTime
    error_code: str | None
    publication_warning_code: str | None


class NexusSnapshot(Closed):
    ordinal: int = Field(gt=0)
    snapshot_json: str


class NexusTurnExport(Closed):
    contract_revision: NexusRevision
    capture_id: NexusUUID
    owner_user_id: NexusUUID
    run_id: NexusUUID
    export_seq: NexusSequence
    retry_of_run_id: NexusUUID | None
    question: NexusMessage
    answer: NexusMessage
    settlement: NexusSettlement
    attachments: tuple[NexusSnapshot, ...]
    citations: tuple[NexusSnapshot, ...]

    @model_validator(mode="after")
    def valid_snapshot(self) -> Self:
        if (
            self.question.message_id == self.answer.message_id
            or int(self.answer.message_seq) != int(self.question.message_seq) + 1
            or self.retry_of_run_id == self.run_id
        ):
            raise ValueError("invalid nexus turn identity")
        for values in (self.attachments, self.citations):
            if len({item.ordinal for item in values}) != len(values):
                raise ValueError("duplicate snapshot ordinal")
        if self.citations != tuple(
            sorted(self.citations, key=lambda item: item.ordinal)
        ):
            raise ValueError("citations must follow publication order")
        for item in (*self.attachments, *self.citations):
            if canonical_json(json.loads(item.snapshot_json)) != item.snapshot_json:
                raise ValueError("evidence json must be canonical")
        return self


def payload_digest(payload: NexusTurnExport) -> str:
    return hashlib.sha256(
        canonical_json(payload.model_dump(mode="json")).encode()
    ).hexdigest()


class NexusActivate(Closed):
    contract_revision: NexusRevision
    capture_id: NexusUUID
    owner_user_id: NexusUUID
    capture_after_message_seq: NexusProgress


class NexusRegistration(Closed):
    capture_id: NexusUUID
    owner_user_id: NexusUUID
    capture_after_message_seq: NexusProgress
    activated_at: NexusTime
    through_export_seq: NexusProgress
    through_export_sha256: NexusDigest | None

    @model_validator(mode="after")
    def progress_pair(self) -> Self:
        if (self.through_export_seq == "0") != (self.through_export_sha256 is None):
            raise ValueError("incomplete nexus progress")
        return self


class NexusState(Closed):
    contract_revision: NexusRevision
    registration: NexusRegistration | None


class NexusIngest(Closed):
    contract_revision: NexusRevision
    expected_through_export_seq: NexusProgress
    payload: NexusTurnExport
    payload_sha256: NexusDigest


class NexusReceipt(Closed):
    capture_id: NexusUUID
    export_seq: NexusSequence
    run_id: NexusUUID
    payload_sha256: NexusDigest


def _registration(row: RowMapping) -> NexusRegistration:
    checkpoint = row["checkpoint_event_id"]
    return NexusRegistration(
        capture_id=row["capture_id"],
        owner_user_id=UUID(row["native_id"]),
        capture_after_message_seq=row["capture_after_event_id"].removeprefix(
            "message-seq:"
        ),
        activated_at=row["captured_from"].astimezone(UTC).isoformat(),
        through_export_seq=checkpoint.removeprefix("export:") if checkpoint else "0",
        through_export_sha256=row["checkpoint_native_digest"],
    )


async def nexus_capture_state(
    connection: AsyncConnection, owner_user_id: UUID
) -> NexusState:
    row = (
        (
            await connection.execute(
                select(source_conversation).where(
                    source_conversation.c.provider == "nexus",
                    source_conversation.c.native_id == str(owner_user_id),
                )
            )
        )
        .mappings()
        .one_or_none()
    )
    return NexusState(
        contract_revision=NEXUS_CONTRACT_REVISION,
        registration=_registration(row) if row is not None else None,
    )


async def register_nexus_source(
    connection: AsyncConnection, request: NexusActivate
) -> NexusRegistration:
    await archive.lock_state(connection)
    state = await nexus_capture_state(connection, request.owner_user_id)
    if state.registration is not None:
        if (
            state.registration.capture_id != request.capture_id
            or state.registration.capture_after_message_seq
            != request.capture_after_message_seq
        ):
            raise MemoryError("source_conflict")
        return state.registration
    row = (
        (
            await connection.execute(
                insert(source_conversation)
                .values(
                    id=uuid4(),
                    provider="nexus",
                    native_id=str(request.owner_user_id),
                    capture_id=request.capture_id,
                    captured_from=datetime.now(UTC),
                    capture_after_event_id=f"message-seq:{request.capture_after_message_seq}",
                )
                .returning(source_conversation)
            )
        )
        .mappings()
        .one()
    )
    return _registration(row)


def _event_ids(payload: NexusTurnExport) -> tuple[str, ...]:
    prefix = f"run:{payload.run_id}"
    return (
        f"{prefix}:question",
        *(f"{prefix}:attachment:{item.ordinal}" for item in payload.attachments),
        f"{prefix}:answer",
        *(f"{prefix}:citation:{item.ordinal}" for item in payload.citations),
    )


def _project(payload: NexusTurnExport, digest: str) -> tuple[Event, ...]:
    event_ids = _event_ids(payload)
    identifiers = iter(event_ids)
    result: list[Event] = []

    def append(
        role: str,
        kind: str,
        text: str,
        attributes: dict[str, JsonValue],
        message: NexusMessage,
        occurred_at: datetime | None,
    ) -> None:
        source = NexusAttributes(
            capture_id=payload.capture_id,
            export_seq=payload.export_seq,
            run_id=payload.run_id,
            retry_of_run_id=payload.retry_of_run_id,
            message_id=message.message_id,
            message_seq=message.message_seq,
            message_created_at=message.created_at,
            turn_status=payload.settlement.status,
            completed_at=payload.settlement.completed_at,
            error_code=payload.settlement.error_code,
            publication_warning_code=payload.settlement.publication_warning_code,
            envelope_sha256=digest,
            event_count=len(event_ids),
            tool=None,
        ).model_dump(mode="json")
        original: dict[str, object] = {
            "native_event_id": next(identifiers),
            "turn_id": str(payload.run_id),
            "native_parent_id": None,
            "role": role,
            "kind": kind,
            "attributes": {**attributes, "nexus": source},
            "text": text,
            "occurred_at": occurred_at.isoformat() if occurred_at else None,
        }
        native_digest = hashlib.sha256(canonical_json(original).encode()).hexdigest()
        original.update(native_digest=native_digest, occurred_at=occurred_at)
        result.append(normalize_event(original))

    append(
        "owner",
        "text",
        payload.question.text,
        {},
        payload.question,
        datetime.fromisoformat(payload.question.created_at),
    )
    for item in payload.attachments:
        append(
            "owner",
            "attachment",
            item.snapshot_json,
            {
                "reference": (
                    f"nexus:{payload.owner_user_id}:run:{payload.run_id}"
                    f":attachment:{item.ordinal}"
                ),
                "media_type": "application/json",
                "text_availability": "full",
            },
            payload.question,
            None,
        )
    append(
        "assistant",
        "text",
        payload.answer.text,
        {},
        payload.answer,
        datetime.fromisoformat(payload.settlement.completed_at),
    )
    for item in payload.citations:
        append(
            "assistant",
            "attachment",
            item.snapshot_json,
            {
                "reference": (
                    f"nexus:{payload.owner_user_id}:run:{payload.run_id}"
                    f":citation:{item.ordinal}"
                ),
                "media_type": "application/json",
                "text_availability": "full",
            },
            payload.answer,
            None,
        )
    return tuple(result)


async def append_nexus_export(
    connection: AsyncConnection,
    request: NexusIngest,
    result_size: retrieval.ResultSize,
) -> NexusReceipt:
    if encoded_size(request.model_dump(mode="json")) > BODY_BYTES:
        raise MemoryError("event_too_large")
    payload = request.payload
    if payload_digest(payload) != request.payload_sha256:
        raise MemoryError("source_conflict")
    state = await archive.lock_state(connection)
    row = (
        (
            await connection.execute(
                select(source_conversation)
                .where(
                    source_conversation.c.provider == "nexus",
                    source_conversation.c.native_id == str(payload.owner_user_id),
                )
                .with_for_update()
            )
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise MemoryError("forbidden")
    registration = _registration(row)
    if registration.capture_id != payload.capture_id or int(
        payload.question.message_seq
    ) < int(registration.capture_after_message_seq):
        raise MemoryError("source_conflict")
    receipt = NexusReceipt(
        capture_id=payload.capture_id,
        export_seq=payload.export_seq,
        run_id=payload.run_id,
        payload_sha256=request.payload_sha256,
    )
    retained = (
        (
            await connection.execute(
                select(source_record, memory_leaf.c.position)
                .outerjoin(
                    memory_leaf, memory_leaf.c.source_record_id == source_record.c.id
                )
                .where(
                    source_record.c.conversation_id == row["id"],
                    source_record.c.turn_id == str(payload.run_id),
                )
                .order_by(source_record.c.sequence)
            )
        )
        .mappings()
        .all()
    )
    if retained:
        expected = _event_ids(payload)
        groups: dict[str, list[RowMapping]] = {}
        for record in retained:
            attributes = cast(dict[str, JsonValue], record["attributes"])
            source = attributes.get("nexus")
            if (
                not isinstance(source, dict)
                or any(
                    source.get(name) != value
                    for name, value in {
                        "capture_id": str(payload.capture_id),
                        "export_seq": payload.export_seq,
                        "run_id": str(payload.run_id),
                        "envelope_sha256": request.payload_sha256,
                        "event_count": len(expected),
                    }.items()
                )
                or record["position"] is None
            ):
                raise MemoryError("source_conflict")
            groups.setdefault(record["native_event_id"], []).append(record)
        if tuple(groups) != expected:
            raise MemoryError("source_conflict")
        for parts in groups.values():
            if any(
                part["part"] != ordinal
                or part["part_count"] != len(parts)
                or part["native_digest"] != parts[0]["native_digest"]
                for ordinal, part in enumerate(parts)
            ):
                raise MemoryError("source_conflict")
        return receipt
    if int(payload.export_seq) <= int(registration.through_export_seq):
        raise MemoryError("source_conflict")
    if request.expected_through_export_seq != registration.through_export_seq:
        raise MemoryError("stale_checkpoint")
    if int(payload.export_seq) != int(registration.through_export_seq) + 1:
        raise MemoryError("source_conflict")
    events = _project(payload, request.payload_sha256)
    await archive.insert_events(
        connection, row, cast(int, state["leaf_count"]), events, result_size
    )
    await connection.execute(
        update(source_conversation)
        .where(source_conversation.c.id == row["id"])
        .values(
            checkpoint_event_id=f"export:{payload.export_seq}",
            checkpoint_native_digest=request.payload_sha256,
            last_read_at=datetime.now(UTC),
        )
    )
    return receipt
