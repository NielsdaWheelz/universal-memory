from __future__ import annotations

from typing import cast

from pgvector.sqlalchemy import VECTOR  # pyright: ignore[reportMissingTypeStubs]
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    and_,
    func,
    literal_column,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.types import TypeEngine

from .policy import EMBEDDING_DIMENSIONS, EVENT_BYTES, NODE_HARD_BYTES, PART_BYTES
from .types import Store

metadata = MetaData(
    naming_convention={
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "ix": "ix_%(table_name)s_%(column_0_name)s",
        "pk": "pk_%(table_name)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
    }
)
vector = cast(TypeEngine[object], VECTOR(EMBEDDING_DIMENSIONS))

# Bind preserved stores without taking ownership of their lifecycle.
memory_log = Table(
    "memory_log",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("text", Text, nullable=False),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column("embedding", vector),
    Column("agent_submission", JSONB),
    Column("synthesis", JSONB),
    CheckConstraint(
        "NOT (agent_submission IS NOT NULL AND synthesis IS NOT NULL)",
        name="provenance_variant",
    ),
)
memory_summary = Table(
    "memory_summary",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("text", Text, nullable=False),
    Column("source_memory_ids", ARRAY(UUID(as_uuid=True)), nullable=False),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column("embedding", vector),
)
memory_lane = Table(
    "memory_lane",
    metadata,
    Column("machine", Text, primary_key=True),
    Column("account", Text, primary_key=True),
    Column("provider", Text, nullable=False),
    Column("activated_at", DateTime(timezone=True), nullable=False),
    Column("last_inventory_at", DateTime(timezone=True)),
    CheckConstraint("provider IN ('codex', 'claude', 'jarvis')", name="provider"),
    CheckConstraint("machine IN ('macbook', 'arch', 'devbox')", name="machine"),
    CheckConstraint(
        "account IN ('codex-personal', 'codex-work', 'codex-work2', "
        "'claude-personal', 'claude-work', 'jarvis')",
        name="account",
    ),
)
source_conversation = Table(
    "source_conversation",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("provider", Text, nullable=False),
    Column("native_id", Text, nullable=False),
    Column("machine", Text, nullable=False),
    Column("account", Text, nullable=False),
    Column("relation", Text),
    Column("parent_native_id", Text),
    Column("working_directory", Text),
    Column("captured_from", DateTime(timezone=True), nullable=False),
    Column("capture_after_event_id", Text),
    Column("checkpoint_event_id", Text),
    Column("checkpoint_native_digest", Text),
    Column("last_read_at", DateTime(timezone=True)),
    Column("capture_error", Text),
    ForeignKeyConstraint(
        ["machine", "account"], ["memory_lane.machine", "memory_lane.account"]
    ),
    UniqueConstraint("provider", "native_id"),
    CheckConstraint("provider IN ('codex', 'claude', 'jarvis')", name="provider"),
    CheckConstraint(
        "relation IS NULL OR relation IN ('child', 'fork')", name="relation"
    ),
    CheckConstraint(
        "(checkpoint_event_id IS NULL) = (checkpoint_native_digest IS NULL)",
        name="checkpoint_pair",
    ),
    CheckConstraint(
        "capture_error IS NULL OR capture_error IN "
        "('native_missing','unsupported','event_too_large','history_changed',"
        "'activation_boundary_lost','source_conflict')",
        name="capture_error",
    ),
    CheckConstraint("octet_length(native_id) BETWEEN 1 AND 256", name="native_id"),
)
source_record = Table(
    "source_record",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "conversation_id",
        UUID(as_uuid=True),
        ForeignKey("source_conversation.id"),
        nullable=False,
    ),
    Column("sequence", BigInteger, nullable=False),
    Column("native_event_id", Text, nullable=False),
    Column("part", Integer, nullable=False),
    Column("part_count", Integer, nullable=False),
    Column("native_digest", Text, nullable=False),
    Column("turn_id", Text),
    Column("native_parent_id", Text),
    Column("role", Text, nullable=False),
    Column("kind", Text, nullable=False),
    Column("attributes", JSONB, nullable=False),
    Column("text", Text, nullable=False),
    Column("occurred_at", DateTime(timezone=True)),
    Column(
        "received_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column("embedding", vector),
    UniqueConstraint("conversation_id", "sequence", name="uq_source_record_sequence"),
    UniqueConstraint(
        "conversation_id",
        "native_event_id",
        "part",
        name="uq_source_record_native_part",
    ),
    CheckConstraint("sequence >= 0 AND part >= 0 AND part < part_count", name="part"),
    CheckConstraint(
        "role IN ('owner','assistant','peer','tool','host','unknown')", name="role"
    ),
    CheckConstraint(
        "kind IN ('text','tool_call','tool_result','attachment','context',"
        "'memory_reference','gap','revision')",
        name="kind",
    ),
    CheckConstraint(f"octet_length(text) <= {PART_BYTES}", name="part_bytes"),
    CheckConstraint(
        "embedding IS NULL OR (kind = 'text' AND role IN ('owner','assistant','peer'))",
        name="embedding_eligible",
    ),
    CheckConstraint(
        "jsonb_typeof(attributes) = 'object' AND "
        f"octet_length(attributes::text) <= {EVENT_BYTES}",
        name="attributes",
    ),
)
memory_leaf = Table(
    "memory_leaf",
    metadata,
    Column("position", BigInteger, primary_key=True),
    Column(
        "source_record_id",
        UUID(as_uuid=True),
        ForeignKey("source_record.id"),
        unique=True,
    ),
    Column("note_id", UUID(as_uuid=True), ForeignKey("memory_log.id"), unique=True),
    CheckConstraint("position >= 0", name="position"),
    CheckConstraint(
        "(source_record_id IS NULL) <> (note_id IS NULL)", name="one_original"
    ),
)
memory_node = Table(
    "memory_node",
    metadata,
    Column("level", Integer, primary_key=True),
    Column("node_index", BigInteger, primary_key=True),
    Column("text", Text, nullable=False),
    CheckConstraint("level BETWEEN 0 AND 62 AND node_index >= 0", name="geometry"),
    CheckConstraint(
        f"length(btrim(text)) > 0 AND octet_length(text) <= {NODE_HARD_BYTES}",
        name="text",
    ),
)
memory_state = Table(
    "memory_state",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("leaf_count", BigInteger, nullable=False, server_default=text("0")),
    Column("view_through", BigInteger, nullable=False, server_default=text("0")),
    Column("main_parts", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("compact_parts", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("main_shrinking", Boolean, nullable=False, server_default=text("false")),
    Column("compact_shrinking", Boolean, nullable=False, server_default=text("false")),
    Column("compaction_error", JSONB),
    Column("dream_through", BigInteger, nullable=False, server_default=text("0")),
    Column("last_nightly_attempt_at", DateTime(timezone=True)),
    CheckConstraint("id = 1", name="singleton"),
    CheckConstraint(
        "0 <= view_through AND view_through <= leaf_count AND "
        "0 <= dream_through AND dream_through <= leaf_count",
        name="progress",
    ),
    CheckConstraint(
        "jsonb_typeof(main_parts) = 'array' AND jsonb_typeof(compact_parts) = 'array'",
        name="frontiers",
    ),
)
MEMORY_TABLES = (
    memory_lane,
    source_conversation,
    source_record,
    memory_leaf,
    memory_node,
    memory_state,
)
RECORD_TABLES: dict[Store, Table] = {
    "source_record": source_record,
    "memory_log": memory_log,
    "memory_summary": memory_summary,
}
conversational_text = and_(
    source_record.c.kind == "text",
    source_record.c.role.in_(("owner", "assistant", "peer")),
)

for table in (memory_log, memory_summary, source_record):
    for language in ("english", "simple"):
        Index(
            f"ix_{table.name}_search_{language}",
            func.to_tsvector(literal_column(f"'{language}'::regconfig"), table.c.text),
            postgresql_using="gin",
            _table=table,
        )
Index("ix_memory_leaf_source", memory_leaf.c.source_record_id)


def guard_statements() -> tuple[str, ...]:
    """DDL composed by the host migration; existing canonical tables are preserved."""
    statements = [
        """
        CREATE OR REPLACE FUNCTION memory_guard_original() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          IF TG_OP IN ('DELETE','TRUNCATE') THEN
            RAISE EXCEPTION 'memory originals are append-only';
          END IF;
          IF TG_OP = 'INSERT' THEN
            IF TG_TABLE_NAME = 'memory_log' THEN
              IF (NEW.agent_submission IS NULL) = (NEW.synthesis IS NULL) THEN
                RAISE EXCEPTION 'new notes require exactly one provenance variant';
              END IF;
            END IF;
            RETURN NEW;
          END IF;
          IF TG_TABLE_NAME = 'memory_lane' THEN
            IF (to_jsonb(NEW) - 'last_inventory_at') IS DISTINCT FROM
               (to_jsonb(OLD) - 'last_inventory_at') THEN
              RAISE EXCEPTION 'memory lane identity is immutable';
            END IF;
          ELSIF TG_TABLE_NAME = 'source_conversation' THEN
            IF (to_jsonb(NEW) - ARRAY['checkpoint_event_id','checkpoint_native_digest',
                 'last_read_at','capture_error']) IS DISTINCT FROM
               (to_jsonb(OLD) - ARRAY['checkpoint_event_id','checkpoint_native_digest',
                 'last_read_at','capture_error']) THEN
              RAISE EXCEPTION 'source conversation identity is immutable';
            END IF;
          ELSIF TG_TABLE_NAME IN ('source_record','memory_log') THEN
            IF (to_jsonb(NEW) - 'embedding') IS DISTINCT FROM
               (to_jsonb(OLD) - 'embedding') THEN
              RAISE EXCEPTION 'memory original is immutable';
            END IF;
          ELSE
            RAISE EXCEPTION 'memory leaf links are immutable';
          END IF;
          RETURN NEW;
        END $$
    """
    ]
    for name in (
        "memory_lane",
        "source_conversation",
        "source_record",
        "memory_leaf",
        "memory_log",
    ):
        statements.extend(
            (
                f"CREATE TRIGGER {name}_universal_guard "
                "BEFORE INSERT OR UPDATE OR DELETE "
                f"ON {name} FOR EACH ROW EXECUTE FUNCTION memory_guard_original()",
                f"CREATE TRIGGER {name}_universal_no_truncate "
                f"BEFORE TRUNCATE ON {name} "
                "FOR EACH STATEMENT EXECUTE FUNCTION memory_guard_original()",
                f"REVOKE DELETE, TRUNCATE ON {name} FROM PUBLIC",
            )
        )
    return tuple(statements)


async def install_guards(connection: AsyncConnection) -> None:
    for statement in guard_statements():
        await connection.execute(text(statement))
