# universal memory

a small postgres-specific python library for retained conversation evidence,
optional authored notes, chronological compression and bounded retrieval. it owns
the archive and tree; the application owns admission, write transactions,
inference, embedding requests, scheduling and transports.

`MemoryStore(read_engine, result_size=..., view_size=...)` serves one corpus. pass
the same bounded read engine and exact transport-result and prompt-view size
functions at composition. `view_size(text)` measures the actual host section,
including escaping and framing; use that same section for main, compactor and
dream views and canonical-json sample text. reads use
read-only transactions with a five-second statement timeout. writes accept an
active `AsyncConnection`; they never commit or open another writer.

- `activate_lane`, `sync_conversations` and `append_events` preserve immutable
  baselines, native identity and atomic checkpoint/leaf allocation.
  before inserting a batch, every proposed part must fit a complete one-record
  open page and leaf zoom within 65,536 bytes, measured as canonical json and by
  the host's `result_size`. the check uses the same record/result builders as
  reads, with the planned uuid and database receipt time. provenance timestamps
  are canonically utc, preserving each original instant. continuation framing
  reserves the longest signed-storage corpus cutoff because that corpus can
  grow; offset and remaining counts use only the immutable event part count or
  the twenty-identity selector bound. a rejected event raises `event_too_large`
  without adding rows, positions or checkpoint progress. after a rejected batch,
  the collector resends its frozen events individually in order, commits the
  valid preceding prefix and parks at the first rejected event. it never crops
  evidence or rereads native history during that recovery.
- `append_note` and `recover_note` accept host-built `NoteProducer`: native
  `Submission(kind="native", ...)` or `NexusSubmission(kind="nexus",
  client="nexus-owner", ...)`. both tags are required. native identities remain
  unchanged; nexus identities include the configured owner uuid. identical retries
  return the original receipt and conflicting reuse fails. nexus notes retain their
  tagged attribution with null native provider/machine/account provenance.
- `next_compaction` produces a complete immutable task. use `free_text` directly
  when present; otherwise supply bounded tool-free inference. `complete_compaction`
  commits the node and persisted frontiers together. the ready queue observes
  committed nodes, so interrupted write transactions remain retryable.
- `main_view` covers a fixed cutoff completely. `view` freezes paged external
  snapshots; `zoom`, `date`, `search` and `open` preserve source provenance and
  enforce that cutoff. the application supplies one query vector to `search`.
- `dream_input` separates eligible new material from prior syntheses and samples
  older original identities. `complete_dream` appends attributed notes and advances
  the physical cursor atomically. `claim_nightly` records one admitted attempt.
- `status`, `retry_capture`, `retry_compression` and `rebuild` support stopped
  maintenance. full rebuild clears derived vectors/tree, preserving originals,
  positions and dream progress.

`schema.MEMORY_TABLES` exports the six added tables. `schema.memory_log` and
`schema.memory_summary` bind the two preserved stores; the host owns their existing
lifecycle. compose the note-provenance columns and `schema.guard_statements()` in
the host migration. initialize `memory_state` with `id=1`. model/provider, host
authority, credentials and http objects do not enter this package.

shared bounds and canonical encoding live in `policy.py`; normalized capture and
closed public values live in `types.py`. `MemoryError.code` is content-free.
`tools.py` owns portable read declarations and the text-only host save contract;
hosts supply execution, identity and grants.
postgres driver and sqlalchemy requirements permit the hosts' compatible major
versions; the library lock keeps its qualified baseline. hosts retain exact pins.
postgres text and jsonb cannot retain nul: normalized events containing it in
retained text or metadata fail as `unsupported`; notes and associations fail as
`invalid_input`. rejection preserves complete events and checkpoint positions.

run `scripts/verify` for lock-checked, frozen static/build checks. the build frontend
and backend run from the locked development environment. the two small postgres
regression groups use a task-owned empty database, explicitly selected with
`MEMORY_TEST_DATABASE_URL`; run them with `uv run python -m unittest discover -s
tests`. their setup recreates memory tables in that database.
