from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal, Self, cast
from uuid import UUID

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from .policy import (
    ARCHIVE_GENERATION,
    BATCH_EVENTS,
    COMPACTION_BATCH_SIZE,
    COMPACTION_READY_WINDOW,
    CURSOR_CHARACTERS,
    DREAM_NOTES,
    DREAM_REFERENCES,
    EVENT_BYTES,
    OPEN_IDENTITIES,
    POSITION_NOTICE,
    RECENT_EXCHANGES,
    SEARCH_QUERY_BYTES,
    SEARCH_QUERY_CHARACTERS,
    MemoryError,
    bounded_text,
    clip_tool_text,
    encoded_size,
    redact_secrets,
    validate_association,
)

type NativeProvider = Literal["codex", "claude", "jarvis"]
type Provider = Literal["codex", "claude", "jarvis", "nexus"]
type Store = Literal["source_record", "memory_log", "memory_summary"]
type Role = Literal["owner", "assistant", "peer", "tool", "host", "unknown"]
type Kind = Literal[
    "text",
    "tool_call",
    "tool_result",
    "attachment",
    "context",
    "memory_reference",
    "gap",
    "revision",
]
type SearchKind = Literal[
    "text",
    "tool_call",
    "tool_result",
    "attachment",
    "context",
    "memory_reference",
    "gap",
    "revision",
    "note",
    "summary",
]


class Closed(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, hide_input_in_errors=True
    )


type Machine = Literal["macbook", "arch", "devbox", "desktop"]
type Account = Literal[
    "codex-personal",
    "codex-work",
    "codex-work2",
    "claude-personal",
    "claude-work",
    "claude-work2",
    "jarvis",
]

type BaselineError = Literal[
    "unavailable",
    "unsupported",
    "event_too_large",
    "history_changed",
    "activation_boundary_lost",
    "source_conflict",
]


class Lane(Closed):
    machine: Machine
    account: Account
    provider: NativeProvider

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (self.account == "jarvis" and self.machine != "devbox") or (
            self.account.split("-", 1)[0] != self.provider
        ):
            raise ValueError("lane provider mismatch")
        return self


class Conversation(Closed):
    native_id: str
    head_event_id: str | None = None
    baseline_error: BaselineError | None = None
    relation: Literal["child", "fork"] | None = None
    parent_native_id: str | None = None
    working_directory: str | None = None
    worker_excluded: bool = False

    @model_validator(mode="after")
    def bounded(self) -> Self:
        if self.baseline_error is not None and self.head_event_id is not None:
            raise ValueError("an unreadable baseline cannot supply a head")
        if self.worker_excluded and (
            self.head_event_id is not None
            or self.baseline_error is not None
            or self.working_directory is not None
        ):
            raise ValueError("worker inventory carries identity and lineage only")
        bounded_text(self.native_id, 256)
        for item in (self.head_event_id, self.parent_native_id):
            if item is not None:
                bounded_text(item, 256)
        if self.working_directory is not None:
            bounded_text(self.working_directory, 4096)
        if self.relation is not None and self.parent_native_id is None:
            raise ValueError("relation needs parent")
        return self


class Origin(Closed):
    native_conversation_id: str
    native_event_id: str
    native_digest: str

    @model_validator(mode="after")
    def bounded(self) -> Self:
        for value in (
            self.native_conversation_id,
            self.native_event_id,
            self.native_digest,
        ):
            bounded_text(value, 256)
        return self


class RecordReference(Closed):
    kind: Literal["record"] = "record"
    store: Store
    id: UUID


class EventIdentity(Closed):
    conversation_id: UUID
    native_event_id: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def bounded_identity(self) -> Self:
        bounded_text(self.native_event_id, 256)
        return self


# Keep the enum inline in tool schemas: their compiler rejects a default beside $ref.
ArchiveGeneration = Literal[1, 2]


def _generation_from_notice(notice: object) -> ArchiveGeneration:
    return ARCHIVE_GENERATION if notice == POSITION_NOTICE else 1


class RangeReference(Closed):
    kind: Literal["range"] = "range"
    generation: ArchiveGeneration = Field(default=1, exclude_if=lambda n: n == 1)
    start: int = Field(ge=0)
    count: int = Field(gt=0)

    @model_validator(mode="after")
    def aligned(self) -> Self:
        if self.count & (self.count - 1) or self.start % self.count:
            raise ValueError("unaligned binary range")
        if self.count > 2**62 or self.start + self.count > 2**63 - 1:
            raise ValueError("range exceeds signed position bound")
        return self


type Reference = Annotated[
    RecordReference | RangeReference, Field(discriminator="kind")
]


def _canonical_uuid(value: object) -> object:
    if isinstance(value, str) and str(UUID(value)) != value:
        raise ValueError("uuid must be canonical")
    return value


def _signed_sequence(value: str) -> str:
    if int(value) > 2**63 - 1:
        raise ValueError("sequence exceeds signed storage bound")
    return value


def _utc_time(value: str) -> str:
    if datetime.fromisoformat(value).utcoffset() != timedelta(0):
        raise ValueError("nexus time must be utc")
    return value


type NexusUUID = Annotated[UUID, BeforeValidator(_canonical_uuid)]
type NexusSequence = Annotated[
    str, Field(pattern=r"^[1-9][0-9]*$"), AfterValidator(_signed_sequence)
]
type NexusProgress = Annotated[
    str, Field(pattern=r"^(0|[1-9][0-9]*)$"), AfterValidator(_signed_sequence)
]
type NexusTime = Annotated[
    str,
    Field(
        pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?(Z|\+00:00)$",
        json_schema_extra={"format": "date-time"},
    ),
    AfterValidator(_utc_time),
]
type NexusStatus = Literal["complete", "error", "cancelled"]
type NexusDigest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class NexusToolAttributes(Closed):
    generation_seq: NexusSequence
    position: int = Field(gt=0)
    status: NexusStatus
    created_at: NexusTime
    reverted_at: NexusTime | None


class NexusAttributes(Closed):
    capture_id: NexusUUID
    export_seq: NexusSequence
    run_id: NexusUUID
    retry_of_run_id: NexusUUID | None
    message_id: NexusUUID | None
    message_seq: NexusSequence | None
    message_created_at: NexusTime | None
    turn_status: NexusStatus
    completed_at: NexusTime
    error_code: str | None
    publication_warning_code: str | None
    envelope_sha256: NexusDigest
    event_count: int = Field(ge=2)
    tool: NexusToolAttributes | None

    @model_validator(mode="after")
    def valid_source(self) -> Self:
        message = (self.message_id, self.message_seq, self.message_created_at)
        if self.tool is None and any(value is None for value in message):
            raise ValueError("nexus message attribution is incomplete")
        if self.tool is not None and any(value is not None for value in message):
            raise ValueError("nexus message/tool attribution mismatch")
        for value in (self.error_code, self.publication_warning_code):
            if value is not None:
                bounded_text(value, 256)
        return self


class TextAttributes(Closed):
    secret_omissions: int = Field(ge=0)
    inherited_origin: Origin | None = None
    nexus: NexusAttributes | None = None


class Delivery(Closed):
    kind: Literal["question", "answer", "answer_note", "report"]
    carrier_event_id: str
    call_id: str | None = None
    delivery_id: str | None = None
    question_id: str | None = None
    sender_id: str | None = None
    ordinal: int = Field(ge=0)

    @model_validator(mode="after")
    def bounded(self) -> Self:
        for value in (
            self.carrier_event_id,
            self.call_id,
            self.delivery_id,
            self.question_id,
            self.sender_id,
        ):
            if value is not None:
                bounded_text(value, 256)
        if self.kind == "report" and self.sender_id is None:
            raise ValueError("report requires an evidenced sender")
        return self


class SpeechAttributes(TextAttributes):
    delivery: Delivery | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class CallAttributes(TextAttributes):
    native_call_id: str
    tool_name: str
    omitted_characters: int | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )


class SourceOutput(Closed):
    state: Literal["reported", "absent", "truncated"]
    omitted_characters: int | None = Field(ge=0)


class ResultAttributes(TextAttributes):
    native_call_id: str
    outcome: Literal["ok", "error", "unknown"]
    child_native_id: str | None = None
    omitted_characters: int = Field(ge=0)
    source_output: SourceOutput | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    exit_code: int | None = Field(default=None, exclude_if=lambda value: value is None)


class AttachmentAttributes(TextAttributes):
    reference: str
    media_type: str
    text_availability: Literal["full", "partial", "none"]


class ContextAttributes(TextAttributes):
    context_kind: Literal["instructions", "environment", "compaction", "other"]
    native_name: str | None = None
    native_source_reference: str | None = None
    body_retention: Literal["reference_only"]


class MemoryAttributes(Closed):
    tool_name: str
    references: tuple[Reference, ...] = Field(max_length=100)
    omitted_reference_count: int = Field(ge=0)
    submission_id: UUID | None = None
    inherited_origin: Origin | None = None
    nexus: NexusAttributes | None = None


class GapAttributes(Closed):
    reason: Literal["unreadable", "unsupported"]
    inherited_origin: Origin | None = None


class RevisionAttributes(Closed):
    reason: Literal["branch"]
    inherited_origin: Origin | None = None


_ATTRIBUTES: dict[str, type[Closed]] = {
    "text": SpeechAttributes,
    "tool_call": CallAttributes,
    "tool_result": ResultAttributes,
    "attachment": AttachmentAttributes,
    "context": ContextAttributes,
    "memory_reference": MemoryAttributes,
    "gap": GapAttributes,
    "revision": RevisionAttributes,
}


class Event(Closed):
    native_event_id: str
    native_digest: str
    turn_id: str | None = None
    native_parent_id: str | None = None
    role: Role
    kind: Kind
    attributes: dict[str, JsonValue]
    text: str
    occurred_at: datetime | None = None

    @model_validator(mode="after")
    def valid_event(self) -> Self:
        pending = list(
            cast(dict[str, JsonValue], self.model_dump(mode="json")).values()
        )
        while pending:
            item = pending.pop()
            if isinstance(item, str) and "\x00" in item:
                raise MemoryError("unsupported")
            if isinstance(item, dict):
                pending.extend(item.keys())
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        for item in (self.native_event_id, self.native_digest):
            bounded_text(item, 256)
        for item in (self.turn_id, self.native_parent_id):
            if item is not None:
                bounded_text(item, 256)
        attributes = _ATTRIBUTES[self.kind].model_validate_json(
            json.dumps(self.attributes)
        )
        if isinstance(attributes, SpeechAttributes) and attributes.delivery is not None:
            speakers = {
                "question": {"assistant", "peer"},
                "answer": {"owner"},
                "answer_note": {"owner"},
                "report": {"peer"},
            }
            if self.role not in speakers[attributes.delivery.kind]:
                raise ValueError("delivery speaker conflicts with its evidence")
        if self.attributes.get("nexus") is not None and (
            self.kind
            not in {
                "text",
                "tool_call",
                "tool_result",
                "attachment",
                "memory_reference",
            }
            or self.attributes.get("inherited_origin") is not None
        ):
            raise ValueError("invalid nexus event kind or native lineage")
        for item in self.attributes.values():
            if isinstance(item, str):
                bounded_text(item, 4096)
        for name in (
            "native_call_id",
            "tool_name",
            "child_native_id",
            "native_name",
            "media_type",
        ):
            item = self.attributes.get(name)
            if isinstance(item, str):
                bounded_text(item, 256)
        if (
            self.kind in {"context", "memory_reference", "gap", "revision"}
            and self.text
        ):
            raise ValueError("reference/structural event has no body")
        if self.occurred_at is not None and self.occurred_at.utcoffset() is None:
            raise ValueError("native time must be timezone aware")
        if encoded_size(self.model_dump(mode="json")) > EVENT_BYTES:
            raise MemoryError("event_too_large")
        return self


def normalize_event(value: Mapping[str, object], *, stock_call: bool = False) -> Event:
    """Apply central retention policy after the provider bound native identity."""
    data = dict(value)
    original_attributes = value.get("attributes")
    if not isinstance(original_attributes, Mapping):
        raise MemoryError("invalid_input")
    attributes = dict(cast(Mapping[str, object], original_attributes))
    text = value.get("text")
    kind = value.get("kind")
    if not isinstance(text, str):
        raise MemoryError("invalid_input")
    if "\x00" in text:
        raise MemoryError("unsupported")
    retained, omissions = redact_secrets(text)
    for name, item in attributes.items():
        if isinstance(item, str):
            attributes[name], count = redact_secrets(item)
            omissions += count
    delivery = attributes.get("delivery")
    if isinstance(delivery, dict):
        retained_delivery = dict(cast(dict[str, object], delivery))
        for name, item in retained_delivery.items():
            if isinstance(item, str):
                retained_delivery[name], count = redact_secrets(item)
                omissions += count
        attributes["delivery"] = retained_delivery
    if kind in {"text", "tool_call", "tool_result", "attachment", "context"}:
        previous = attributes.get("secret_omissions", 0)
        if type(previous) is not int or previous < 0:
            raise MemoryError("invalid_input")
        attributes["secret_omissions"] = previous + omissions
    trim_arguments = kind == "tool_call" and stock_call
    if kind == "tool_result" or trim_arguments:
        retained, omitted = clip_tool_text(retained, arguments=trim_arguments)
        previous = attributes.get("omitted_characters", 0)
        if type(previous) is not int or previous < 0:
            raise MemoryError("invalid_input")
        attributes["omitted_characters"] = previous + omitted
    if kind in {"context", "memory_reference", "gap", "revision"}:
        retained = ""
    data["text"] = retained
    data["attributes"] = attributes
    try:
        return Event.model_validate(data)
    except ValidationError as error:
        for issue in error.errors(include_input=False):
            cause = issue.get("ctx", {}).get("error")
            if isinstance(cause, MemoryError) and cause.code in {
                "event_too_large",
                "unsupported",
            }:
                raise cause from None
        raise


type CaptureDigest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Occurrence(Closed):
    native_event_id: str
    native_digest: str
    capture_digest: CaptureDigest
    records: tuple[Event, ...] = Field(max_length=BATCH_EVENTS)

    @model_validator(mode="after")
    def evidence(self) -> Self:
        bounded_text(self.native_event_id, 256)
        bounded_text(self.native_digest, 256)
        for record in self.records:
            if record.role == "tool" or record.kind not in {
                "text",
                "attachment",
                "context",
                "gap",
                "revision",
            }:
                raise ValueError("native capture excludes ordinary tool records")
            if record.attributes.get("nexus") is not None:
                raise ValueError("native capture excludes nexus records")
            delivery = record.attributes.get("delivery")
            if delivery is None:
                if (record.native_event_id, record.native_digest) != (
                    self.native_event_id,
                    self.native_digest,
                ):
                    raise ValueError("ordinary records must keep their source identity")
                continue
            assert isinstance(delivery, dict)
            if (
                delivery["carrier_event_id"] != self.native_event_id
                or record.attributes.get("inherited_origin") is not None
                or not record.native_event_id.startswith("delivery:")
                or len(record.native_event_id) != 73
                or any(
                    char not in "0123456789abcdef"
                    for char in record.native_event_id[9:]
                )
            ):
                raise ValueError("derived speech requires its current delivery carrier")
        return self


def memory_reference_attributes(tool_name: str, value: object) -> dict[str, JsonValue]:
    """Extract identities only; malformed save inputs never acquire a text body."""
    bounded_text(tool_name, 256)
    found: dict[Reference, None] = {}
    pending: list[tuple[object, ArchiveGeneration]] = [(value, 1)]
    submission: UUID | None = None
    while pending:
        item, generation = pending.pop()
        if isinstance(item, Closed):
            item = item.model_dump(mode="json")
        if isinstance(item, dict):
            fields = cast(dict[str, object], item)
            store, identifier = fields.get("store"), fields.get("id")
            if (
                isinstance(store, str)
                and store
                in {
                    "source_record",
                    "memory_log",
                    "memory_summary",
                }
                and isinstance(identifier, str)
            ):
                try:
                    found[
                        RecordReference(store=cast(Store, store), id=UUID(identifier))
                    ] = None
                except ValueError:
                    pass
            start, count = fields.get("start"), fields.get("count")
            if type(start) is int and type(count) is int:
                try:
                    found[
                        RangeReference.model_validate(
                            {
                                "start": start,
                                "count": count,
                                "generation": fields.get(
                                    "generation",
                                    generation
                                    if isinstance(fields.get("text"), str)
                                    else 1,
                                ),
                            }
                        )
                    ] = None
                except ValueError:
                    pass
            key = fields.get("submission_id")
            if tool_name in {"memory_save_note", "memory.save_note"} and isinstance(
                key, str
            ):
                try:
                    submission = UUID(key)
                except ValueError:
                    pass
            pending.extend(
                (
                    child,
                    _generation_from_notice(fields.get("notice"))
                    if key == "nodes"
                    else 1,
                )
                for key, child in fields.items()
            )
        elif isinstance(item, list | tuple):
            pending.extend(
                (child, generation) for child in cast(Sequence[object], item)
            )
    references = list(found)
    result: dict[str, JsonValue] = {
        "tool_name": tool_name,
        "references": [item.model_dump(mode="json") for item in references[:100]],
        "omitted_reference_count": max(0, len(references) - 100),
    }
    if submission is not None:
        result["submission_id"] = str(submission)
    return result


class Submission(Lane):
    kind: Literal["native"]
    submission_id: UUID
    native_conversation_id: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def valid_association(self) -> Self:
        validate_association(self.native_conversation_id)
        return self


class NexusSubmission(Closed):
    kind: Literal["nexus"]
    client: Literal["nexus-owner"]
    owner_user_id: UUID
    submission_id: UUID
    native_conversation_id: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def valid_association(self) -> Self:
        validate_association(self.native_conversation_id)
        return self


type NoteProducer = Annotated[Submission | NexusSubmission, Field(discriminator="kind")]


class NoteReceipt(Closed):
    store: Literal["memory_log"] = "memory_log"
    id: UUID
    created_at: datetime
    native_conversation_id: str | None


class LaneReceipt(Closed):
    lane: Lane
    activated_at: datetime
    last_inventory_at: datetime | None


class ConversationReceipt(Closed):
    id: UUID
    provider: Provider
    native_id: str
    machine: str | None
    account: str | None
    capture_after_event_id: str | None
    baseline_error: BaselineError | None
    checkpoint_event_id: str | None
    checkpoint_native_digest: str | None
    checkpoint_capture_digest: CaptureDigest | None = None
    capture_generation: int = Field(ge=0)
    worker_excluded: bool = False
    last_read_at: datetime | None
    capture_error: str | None
    foreign: bool = False


class Node(Closed):
    start: int
    count: int
    text: str

    @property
    def reference(self) -> RangeReference:
        return self.reference_for(1)

    def reference_for(self, generation: ArchiveGeneration) -> RangeReference:
        return RangeReference(generation=generation, start=self.start, count=self.count)


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


type SourceTimestamp = Annotated[AwareDatetime, AfterValidator(_as_utc)]


class Fragment(Closed):
    ref: RecordReference
    part: int = Field(ge=0)
    byte_offset: int = Field(ge=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def retained_part(self) -> Self:
        if self.ref.store != "source_record":
            raise ValueError("an excerpt cites source records")
        return self


class Excerpt(Closed):
    event: EventIdentity
    first_position: int = Field(ge=0)
    part_count: int = Field(gt=0)
    role: Literal["owner", "assistant", "peer"]
    provider: Provider
    machine: Machine | None
    account: Account | None
    occurred_at: SourceTimestamp | None
    received_at: SourceTimestamp
    fragments: tuple[Fragment, ...] = Field(min_length=1)
    omitted_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered_fragments(self) -> Self:
        previous: Fragment | None = None
        for fragment in self.fragments:
            if fragment.part >= self.part_count or (
                previous is not None
                and (
                    fragment.part < previous.part
                    or (
                        fragment.part == previous.part
                        and (
                            fragment.ref != previous.ref
                            or fragment.byte_offset
                            <= previous.byte_offset + len(previous.text.encode())
                        )
                    )
                )
            ):
                raise ValueError("excerpt fragments must be ordered and disjoint")
            previous = fragment
        return self


class Exchange(Closed):
    owner: tuple[Excerpt, ...] = Field(min_length=1)
    omitted_owner_events: int = Field(ge=0)
    preceding: Excerpt | None
    following: Excerpt | None

    @model_validator(mode="after")
    def conversation_order(self) -> Self:
        if any(excerpt.role != "owner" for excerpt in self.owner) or any(
            excerpt is not None and excerpt.role == "owner"
            for excerpt in (self.preceding, self.following)
        ):
            raise ValueError("an exchange separates owner and neighboring messages")
        end = 0
        for excerpt in (self.preceding, *self.owner, self.following):
            if excerpt is None:
                continue
            if (
                excerpt.first_position < end
                or excerpt.event.conversation_id != self.owner[0].event.conversation_id
            ):
                raise ValueError("an exchange stays in conversation order")
            end = excerpt.first_position + excerpt.part_count
        return self


def _view_end(
    start: int,
    through: int,
    admitted: int,
    nodes: tuple[Node, ...],
    recent: tuple[Exchange, ...],
) -> int:
    if not 0 <= start <= through <= admitted <= 2**63 - 1:
        raise ValueError("invalid memory coverage")
    for node in nodes:
        if node.reference.start != start:
            raise ValueError("memory prefix is discontinuous")
        start += node.count
    if start > through:
        raise ValueError("memory prefix exceeds coverage")
    owners: set[EventIdentity] = set()
    for exchange in recent:
        for excerpt in (exchange.preceding, *exchange.owner, exchange.following):
            if excerpt is None:
                continue
            if excerpt.first_position + excerpt.part_count > admitted:
                raise ValueError("recent evidence exceeds admitted coverage")
            if excerpt.role == "owner":
                if excerpt.event in owners:
                    raise ValueError("recent owner evidence is repeated")
                owners.add(excerpt.event)
    return start


class ReadyView(Closed):
    generation: ArchiveGeneration = Field(default=1, exclude_if=lambda n: n == 1)
    nodes: tuple[Node, ...]
    through: int
    admitted_through: int
    recent: tuple[Exchange, ...] = Field(max_length=RECENT_EXCHANGES)

    @model_validator(mode="after")
    def complete_prefix(self) -> Self:
        if _view_end(
            0, self.through, self.admitted_through, self.nodes, self.recent
        ) != (self.through):
            raise ValueError("memory prefix is incomplete")
        return self


class CompactionTask(Closed):
    start: int
    count: int
    source: str


class CompactionBatch(Closed):
    tasks: tuple[CompactionTask, ...] = Field(max_length=COMPACTION_BATCH_SIZE)
    copies: tuple[Node, ...] = Field(max_length=COMPACTION_READY_WINDOW)
    context: str
    context_through: int

    @model_validator(mode="after")
    def nonempty(self) -> Self:
        if not self.tasks and not self.copies:
            raise ValueError("compaction batch is empty")
        return self


class Sample(Closed):
    store: Store
    id: UUID
    nodes: tuple[Node, ...]
    partial_coverage: bool


class DreamInput(Closed):
    generation: ArchiveGeneration = Field(default=1, exclude_if=lambda n: n == 1)
    seed_start: int
    seed_end: int
    cutoff: int
    nodes: tuple[Node, ...]
    sample: tuple[Sample, ...]
    references: tuple[Reference, ...]


class DreamNote(Closed):
    text: str
    references: tuple[Reference, ...] = Field(min_length=1, max_length=DREAM_REFERENCES)


class DreamResult(Closed):
    notes: tuple[DreamNote, ...] = Field(max_length=DREAM_NOTES)


class ViewPage(Closed):
    @property
    def generation(self) -> ArchiveGeneration:
        return _generation_from_notice(self.notice)

    notice: str
    nodes: tuple[Node, ...]
    through: int
    admitted_through: int
    recent: tuple[Exchange, ...] = Field(max_length=RECENT_EXCHANGES)
    cursor: str | None = Field(max_length=CURSOR_CHARACTERS)

    @model_validator(mode="after")
    def continuation(self) -> Self:
        if self.cursor is not None and not (self.nodes or self.recent):
            raise ValueError("memory continuation has no items")
        _view_end(
            self.nodes[0].start if self.nodes else 0,
            self.through,
            self.admitted_through,
            self.nodes,
            self.recent,
        )
        return self


class SearchInput(Closed):
    query: str = Field(min_length=1, max_length=SEARCH_QUERY_CHARACTERS)
    stores: tuple[Store, ...] = Field(
        default=("source_record", "memory_log", "memory_summary"),
        min_length=1,
        max_length=3,
    )
    kinds: tuple[SearchKind, ...] | None = Field(default=None, max_length=10)
    limit: int = Field(default=10, ge=1, le=20)

    @model_validator(mode="after")
    def valid_query(self) -> Self:
        bounded_text(self.query, SEARCH_QUERY_BYTES)
        if len(set(self.stores)) != len(self.stores) or (
            self.kinds is not None and len(set(self.kinds)) != len(self.kinds)
        ):
            raise ValueError("invalid search input")
        return self


class ViewInput(Closed):
    cursor: str | None = Field(default=None, max_length=CURSOR_CHARACTERS)


class ZoomInput(Closed):
    generation: Literal[1, 2] = 1
    start: int = Field(ge=0)
    count: int = Field(gt=0)

    @model_validator(mode="after")
    def aligned(self) -> Self:
        RangeReference(start=self.start, count=self.count)
        return self


class DateInput(Closed):
    generation: Literal[1, 2] = 1
    position: int = Field(ge=0)

    @model_validator(mode="after")
    def signed_position(self) -> Self:
        if self.position > 2**63 - 1:
            raise ValueError("position exceeds signed storage bound")
        return self


class OpenInput(Closed):
    identities: tuple[RecordReference, ...] = Field(
        default=(), max_length=OPEN_IDENTITIES
    )
    event: EventIdentity | None = None
    cursor: str | None = Field(default=None, max_length=CURSOR_CHARACTERS)

    @model_validator(mode="after")
    def one_selector(self) -> Self:
        if bool(self.identities) == (self.event is not None):
            raise ValueError("open needs identities or an event")
        if len(set(self.identities)) != len(self.identities):
            raise ValueError("duplicate identities")
        return self


class Synthesis(Closed):
    generation: ArchiveGeneration = Field(default=1, exclude_if=lambda n: n == 1)
    seed_start: int = Field(ge=0)
    seed_end: int = Field(ge=0)
    references: tuple[Reference, ...] = Field(min_length=1, max_length=DREAM_REFERENCES)


class Provenance(Closed):
    provider: Provider | None = None
    machine: str | None = None
    account: str | None = None
    conversation_id: UUID | None = None
    native_conversation_id: str | None = None
    occurred_at: datetime | None = None
    received_at: datetime | None = None
    created_at: datetime | None = None
    agent_submission: NoteProducer | None = None
    synthesis: Synthesis | None = None
    legacy: bool = False

    @field_validator("occurred_at", "received_at", "created_at")
    @classmethod
    def utc_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.utcoffset() is None:
            raise ValueError("provenance time must be timezone aware")
        return _as_utc(value)


class RecordBase(Closed):
    store: Store
    id: UUID
    role: Literal["owner", "assistant", "peer", "tool", "host", "unknown", "agent"]
    text: str
    provenance: Provenance
    native_event_id: str | None = None
    native_digest: str | None = None
    turn_id: str | None = None
    native_parent_id: str | None = None
    part: int | None = None
    part_count: int | None = None
    source_memory_ids: tuple[UUID, ...] = ()
    clipped: bool = False

    @model_serializer(mode="wrap")
    def complete_read_result(
        self, serialize: SerializerFunctionWrapHandler
    ) -> dict[str, JsonValue]:
        # Stored event attributes and historical citations retain their sparse
        # encoding. A public record implements the closed tool result schema.
        wire = serialize(self)
        attributes = wire["attributes"]
        match wire["kind"]:
            case "text":
                attributes.setdefault("delivery", None)
            case "tool_call":
                attributes.setdefault("omitted_characters", None)
            case "tool_result":
                attributes.setdefault("source_output", None)
                attributes.setdefault("exit_code", None)
            case "memory_reference":
                for ref in attributes["references"]:
                    if ref["kind"] == "range":
                        ref.setdefault("generation", 1)
            case _:
                pass
        synthesis = wire["provenance"]["synthesis"]
        if synthesis is not None:
            synthesis.setdefault("generation", 1)
            for ref in synthesis["references"]:
                if ref["kind"] == "range":
                    ref.setdefault("generation", 1)
        return wire

    @property
    def reference(self) -> RecordReference:
        return RecordReference(store=self.store, id=self.id)


class TextRecord(RecordBase):
    kind: Literal["text"]
    attributes: SpeechAttributes


class CallRecord(RecordBase):
    kind: Literal["tool_call"]
    attributes: CallAttributes


class ResultRecord(RecordBase):
    kind: Literal["tool_result"]
    attributes: ResultAttributes


class AttachmentRecord(RecordBase):
    kind: Literal["attachment"]
    attributes: AttachmentAttributes


class ContextRecord(RecordBase):
    kind: Literal["context"]
    attributes: ContextAttributes


class MemoryRecord(RecordBase):
    kind: Literal["memory_reference"]
    attributes: MemoryAttributes


class GapRecord(RecordBase):
    kind: Literal["gap"]
    attributes: GapAttributes


class RevisionRecord(RecordBase):
    kind: Literal["revision"]
    attributes: RevisionAttributes


class NoteRecord(RecordBase):
    kind: Literal["note"]
    attributes: None = None


class SummaryRecord(RecordBase):
    kind: Literal["summary"]
    attributes: None = None


type Record = Annotated[
    TextRecord
    | CallRecord
    | ResultRecord
    | AttachmentRecord
    | ContextRecord
    | MemoryRecord
    | GapRecord
    | RevisionRecord
    | NoteRecord
    | SummaryRecord,
    Field(discriminator="kind"),
]
_RECORD: TypeAdapter[Record] = TypeAdapter(Record)


def build_record(**values: object) -> Record:
    attributes = values.get("attributes")
    if attributes is not None:
        values["attributes"] = _ATTRIBUTES[
            cast(str, values["kind"])
        ].model_validate_json(json.dumps(attributes))
    return _RECORD.validate_python(values)


class Missing(Closed):
    kind: Literal["missing"] = "missing"
    store: Store
    id: UUID
    error: Literal["not_found", "retired"] = "not_found"


type OpenRecord = Annotated[Record | Missing, Field(discriminator="kind")]


class OpenPage(Closed):
    notice: str
    records: tuple[OpenRecord, ...]
    cursor: str | None
    remaining: int


class SearchResult(Closed):
    notice: str
    records: tuple[Record, ...]
    lexical_candidates: int
    semantic_candidates: int
    candidates: int
    returned: int
    omitted_limit: int
    omitted_bytes: int
    missing_embeddings: int


class ZoomResult(Closed):
    @property
    def generation(self) -> ArchiveGeneration:
        return _generation_from_notice(self.notice)

    notice: str
    nodes: tuple[Node, ...] = ()
    record: Record | None = None
    full_event: EventIdentity | None = None


class DateResult(Closed):
    notice: str
    position: int
    occurred_at: datetime | None
    received_at: datetime
    note_created_at: bool
