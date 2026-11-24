# universal memory

a small postgres-specific python library for retained conversation evidence,
optional authored notes, chronological compression and bounded retrieval. it owns
the archive and tree; the application owns admission, write transactions,
inference, embedding requests, scheduling and transports.

`MemoryStore(read_engine, result_size=..., view_size=...)` serves one corpus. pass
the same bounded read engine and exact transport-result and prompt-view size
functions at composition. `view_size(text)` measures the actual host section,
including escaping and framing; use that same section for main, compactor and
dream views and `llm_tools.prompt_json_text(sample_values)`. present those samples
as `PromptJson(sample_values)` under that section. reads use
read-only transactions with a five-second statement timeout. writes accept an
active `AsyncConnection`; they never commit or open another writer.

`render_view` and `render_dream_view` present each node as
`start+count|<json string>`, using `llm_tools.prompt_json_text` to preserve raw
text reversibly. originals and structured api values stay unencoded. a constant
historical-evidence title precedes the nodes; changing coverage or interval
metadata follows them. appending nodes preserves the earlier node prefix.

- `activate_lane`, `sync_conversations` and `append_events` preserve immutable
  baselines, native identity and atomic checkpoint/leaf allocation.
  activation records every identity in the complete inventory: readable heads
  establish their cuts, and `baseline_error` retains unreadable identities as
  pending. a null head without an error is a known empty prefix. pending rows
  accept neither events nor ordinary capture repair. `resolve_baseline` records
  another read error or establishes the first successful head and observation
  time; retries return an established cut unchanged. material before that later
  cut remains excluded. ordinary discovery carries neither head nor baseline
  error: newly discovered conversations capture from their beginning. status
  exposes pending baseline errors separately from post-activation capture errors.
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
- nexus capture is a separate account-owned source, not a native lane.
  `register_nexus_source(connection, NexusActivate)` fixes its capture id and
  inclusive first question sequence; an exact retry returns the same registration.
  `nexus_capture_state(connection, owner_user_id)` returns that registration and
  committed export progress. `append_nexus_export(connection, NexusIngest)` commits
  all projected events, source parts, leaf positions and export progress together.
  the caller commits before acknowledging; the library never owns delivery.
  exact old exports return their original receipt from complete retained evidence,
  before applying current new-content bounds. changed identities or content fail.
  the source retains its frozen pending export until acknowledgement, including
  after deleting the conversation. no edit/delete sync or historical import.
  `nexus.py` owns the closed values and canonical payload digest; local values
  have no transport size cap. central append bounds the complete ingest request
  at 16 mib, with the same per-event, part and complete-open guarantees as native
  capture. explicit nexus provenance has no fabricated native machine/account.
  UTC timestamp strings retain their original spelling for digest/replay; database
  occurrence times parse the same instants. tree presentation omits only capture
  id, export sequence, envelope digest and event count from nexus metadata;
  original records and public reads retain them. tool records are stored receipts,
  not reconstructed transient outputs.
- `next_compaction` produces one immutable `CompactionBatch`: up to eight paid
  `tasks`, up to 128 deterministic `copies`, and one bounded completed-prefix
  context ending before the earliest paid source. complete sources at most 2,048
  utf-8 bytes are copied byte-exact, including whitespace; they consume no paid
  slots. every parent uses already committed children. a free-only batch is
  progress; `None` means no eligible work.
  `batch.model_input(size_feedback)` supplies the plain inference packet; hosts
  pack initial and retry subsets using their actual complete request bound, without
  clipping sources. batch siblings are visible together, but each addressed summary
  must remain supported by its own source. the host owns that content contract.
  `complete_compaction(connection, batch, results)` accepts only paid addressed
  `Node` results. it validates the entire identity set, sources and byte bounds
  before inserting anything, then writes all copies/results and both frontiers
  together in the caller's transaction. inferred prose is stripped; copies remain
  exact. the ready cache learns only from later committed reads, so rollback cannot
  enable a parent. no singleton compaction interface or durable job state remains.
- `ready_view` returns one immutable value with the complete ready prefix and
  up to four attributed recent owner exchanges, independent of inference.
  `render_ready_view` and `render_nexus_ready_view` expose the actual prefix
  and admission cutoffs. fitting accounts for both rendered envelopes.
  `view` pages whole nodes and exchanges with the disposable cursor, and
  `collect_ready_view` validates a complete sequence. `main_view` prepares
  complete fixed-cutoff coverage for maintenance. `zoom`, `date`, `search`
  and `open` enforce the admitted cutoff independently of prefix coverage.
  use an excerpt's event identity to reopen every retained source part.
  after historical cleanup, numeric positions use archive generation 2. pass the
  generation shown in the view to `zoom` and `date`. unqualified older
  coordinates return `retired_range`. a node alone has no current generation.
  use `node.reference_for(page.generation)` on a view or zoom page. when
  rendering a dream, pass `generation=seed.generation` to `render_dream_view`.
  paged opens from the prior generation
  return `stale_view`. surviving record and event identities still open. removed
  records and events return `retired` from a body-free address table outside
  the tree and search. saved synthesis retains the generation of its evidence.
- `embedding_candidates` selects ordered eligible records with null vectors.
  `write_embeddings` validates a batch and writes it in the caller's
  transaction. the host owns leadership, provider dispatch and retry.
  `search` takes one query vector and reads ranks and coverage in one snapshot.
- `dream_input` selects eligible new seeds only through committed ready coverage,
  retaining its full admitted cutoff for reads and references. unbuilt/unselected
  seeds stay pending; no new ready interval returns no seed or sample. it excludes
  prior syntheses from seeds and samples older original identities.
  `complete_dream` appends attributed notes and advances the physical cursor
  atomically. `claim_nightly` records one admitted attempt.
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
`types.Account` owns the fixed native account vocabulary, including
`claude-work2`, and supplies the database account constraint.
`tools.py` owns portable read declarations and the text-only host save contract;
hosts supply execution, identity and grants.

`universal_memory.tools.ORIENTATION_GUIDANCE` is the shared prompt for hosts that
supply a ready prefix and recent exchanges. start there, zoom relevant discussions, and use the
existing keyword/semantic search when no useful route remains. remembered user
preferences and decisions are default working knowledge, preserving corrections,
attribution and uncertainty without granting permission. leaf `full_event` feeds
open's `event` selector; follow all cursor pages for complete retained originals,
keeping source omissions explicit.

`NATIVE_ORIENTATION_GUIDANCE` includes that guidance, native startup/resume,
compaction/inherited-child boundaries, outages and useful nonredundant notes.
mcp servers serve it as instructions; fleet configuration may check in its exact
text plus a terminal newline, recording and verifying the immutable source pin.
the native export leads with startup: read every page of one frozen ready snapshot
unless usable completed orientation is supplied or inherited. this shared startup
step is separate from optional provider-local recall and relevance-based zoom/search.
necessary tool discovery and instruction/bootstrap reads may precede it; task-file
reads/actions wait for complete orientation or established memory unavailability,
including when tools can run concurrently. this clarifies prompt order; it does
not establish reliable model adherence or override higher-priority instructions.
preserve ready/admitted coverage and unknown capture
lag. restart stale cursors without splicing snapshots. external clients may
continue independent work during outages with material gaps explicit; host-required
context preparation retains the host's failure policy. no new grants or manual
tree compaction. isolated host roles retain their inputs and zero-tool completion.
external note retries reuse the original `submission_id` and identical arguments,
including the conversation association; a missing receipt can follow a committed
save. host-bound note tools keep identity/recovery with the host and accept text.

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
