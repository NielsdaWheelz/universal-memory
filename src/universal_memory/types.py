from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Annotated, Literal, Self, cast
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from .policy import (
    CURSOR_CHARACTERS,
    DREAM_NOTES,
    DREAM_REFERENCES,
    EVENT_BYTES,
    NODE_TARGET_BYTES,
    OPEN_IDENTITIES,
    SEARCH_QUERY_BYTES,
    SEARCH_QUERY_CHARACTERS,
    MemoryError,
    bounded_text,
    clip_tool_text,
    encoded_size,
    redact_secrets,
    validate_association,
)

type Provider = Literal["codex", "claude", "jarvis"]
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


class Lane(Closed):
    machine: Literal["macbook", "arch", "devbox"]
    account: Literal[
        "codex-personal",
        "codex-work",
        "codex-work2",
        "claude-personal",
        "claude-work",
        "jarvis",
    ]
    provider: Provider

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
    relation: Literal["child", "fork"] | None = None
    parent_native_id: str | None = None
    working_directory: str | None = None

    @model_validator(mode="after")
    def bounded(self) -> Self:
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


class RangeReference(Closed):
    kind: Literal["range"] = "range"
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


class TextAttributes(Closed):
    secret_omissions: int = Field(ge=0)
    inherited_origin: Origin | None = None


class CallAttributes(TextAttributes):
    native_call_id: str
    tool_name: str


class ResultAttributes(TextAttributes):
    native_call_id: str
    outcome: Literal["ok", "error", "unknown"]
    child_native_id: str | None = None
    omitted_characters: int = Field(ge=0)


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


class GapAttributes(Closed):
    reason: Literal["unreadable", "unsupported"]
    inherited_origin: Origin | None = None


class RevisionAttributes(Closed):
    reason: Literal["branch"]
    inherited_origin: Origin | None = None


_ATTRIBUTES: dict[str, type[Closed]] = {
    "text": TextAttributes,
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
        _ATTRIBUTES[self.kind].model_validate_json(json.dumps(self.attributes))
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


def normalize_event(value: Mapping[str, object]) -> Event:
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
    retained, omissions = redact_secrets(text)
    for name, item in attributes.items():
        if isinstance(item, str):
            attributes[name], count = redact_secrets(item)
            omissions += count
    if kind in {"text", "tool_call", "tool_result", "attachment", "context"}:
        previous = attributes.get("secret_omissions", 0)
        if type(previous) is not int or previous < 0:
            raise MemoryError("invalid_input")
        attributes["secret_omissions"] = previous + omissions
    if kind == "tool_result":
        retained, omitted = clip_tool_text(retained)
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


def memory_reference_attributes(tool_name: str, value: object) -> dict[str, JsonValue]:
    """Extract identities only; malformed save inputs never acquire a text body."""
    bounded_text(tool_name, 256)
    found: dict[Reference, None] = {}
    pending = [value]
    submission: UUID | None = None
    while pending:
        item = pending.pop()
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
                    found[RangeReference(start=start, count=count)] = None
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
            pending.extend(fields.values())
        elif isinstance(item, list | tuple):
            pending.extend(cast(Sequence[object], item))
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
    machine: str
    account: str
    capture_after_event_id: str | None
    checkpoint_event_id: str | None
    checkpoint_native_digest: str | None
    last_read_at: datetime | None
    capture_error: str | None
    foreign: bool = False


class Node(Closed):
    start: int
    count: int
    text: str

    @property
    def reference(self) -> RangeReference:
        return RangeReference(start=self.start, count=self.count)


class CompactionTask(Closed):
    start: int
    count: int
    source: str
    context: str
    context_through: int
    free_text: str | None = None
    ruler: str = "-" * NODE_TARGET_BYTES


class Sample(Closed):
    store: Store
    id: UUID
    nodes: tuple[Node, ...]
    partial_coverage: bool


class DreamInput(Closed):
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
    notice: str
    nodes: tuple[Node, ...]
    through: int
    admitted_through: int
    cursor: str | None


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
    start: int = Field(ge=0)
    count: int = Field(gt=0)

    @model_validator(mode="after")
    def aligned(self) -> Self:
        RangeReference(start=self.start, count=self.count)
        return self


class DateInput(Closed):
    position: int = Field(ge=0)

    @model_validator(mode="after")
    def signed_position(self) -> Self:
        if self.position > 2**63 - 1:
            raise ValueError("position exceeds signed storage bound")
        return self


class EventIdentity(Closed):
    conversation_id: UUID
    native_event_id: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def bounded_identity(self) -> Self:
        bounded_text(self.native_event_id, 256)
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
        return value.astimezone(UTC)


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

    @property
    def reference(self) -> RecordReference:
        return RecordReference(store=self.store, id=self.id)


class TextRecord(RecordBase):
    kind: Literal["text"]
    attributes: TextAttributes


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
    error: Literal["not_found"] = "not_found"


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
