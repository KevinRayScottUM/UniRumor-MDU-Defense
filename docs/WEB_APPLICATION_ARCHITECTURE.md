# Web Application Architecture Contract

Status: **Frozen for Task07 implementation**

Applies to: the UniRumor MDU Defense thesis/demo/research web application

API namespace: `/api/v1`

This document is an implementation contract, not an implementation. Task07A-1
creates no React, TypeScript, FastAPI, server, route, queue, or model code.

## 1. Scope and non-goals

### Scope

Task07 will add a dedicated static React frontend and a thin FastAPI web layer
around the closed Task06 production boundary. The baseline is deliberately
small enough for one DICC GPU node while remaining safe for a public thesis
demonstration:

- React + TypeScript + Vite static frontend;
- FastAPI JSON/upload API at `/api/v1`;
- one bounded in-memory job manager;
- one GPU execution worker;
- one long-lived `ProductionExecutionService` and `ProductionRuntime` per
  server process;
- polling as the complete status-delivery mechanism;
- Cloudflare Pages for static presentation and Cloudflare Tunnel for API
  ingress when institutional policy allows it; and
- a dedicated web runtime workspace inside the configured runtime cache root.

This is a single-node thesis/demo/research deployment. It is not represented as
a horizontally scaled commercial SaaS system.

### Non-goals

The baseline does not:

- create, duplicate, or modify an inference pipeline;
- shell out to `app.production_cli`;
- call `FrozenG1Runner`, `VideoMultimodalRunner`, Whisper, PaddleOCR, SigLIP,
  or Qwen directly;
- rebuild `ProductionResult`, recalculate probabilities, reinterpret logits,
  add thresholds, or change evidence sufficiency;
- expose scientific constants as web configuration;
- use a long-held `POST /predict` request;
- require WebSocket or server-sent events (SSE);
- require Redis, Celery, Kafka, RabbitMQ, PostgreSQL, Kubernetes, Docker Swarm,
  object storage, CDN video storage, or an external telemetry SaaS;
- use a rented AWS, GCP, Azure, or other cloud GPU as the baseline;
- use Gradio or Streamlit as the primary public or thesis-defense interface; or
- access Validation/Test data or write into source dataset directories.

Gradio may only be considered later as an isolated developer/debug surface. It
must not become the primary frontend or an alternate inference composition.

## 2. Decision classes

### Frozen decisions

| Concern | Baseline decision | Rationale |
| --- | --- | --- |
| Production integration | Call `ProductionExecutionService.execute()` directly | It owns Task06 runtime and packaging failure semantics. |
| API server | FastAPI | It provides a typed Python HTTP boundary without changing the production graph. |
| Frontend | React + TypeScript + Vite static build | It produces an independent, polished frontend deployable through Cloudflare Pages. |
| Job delivery | Polling | It survives multi-minute inference without a permanently open inference request. |
| Process model | One authorized API process and one worker | It preserves in-memory ownership and makes GPU concurrency enforceable. |
| Process singleton / server lock | One active Task07 API/runtime process per configured `WEB_RUNTIME_ROOT`, enforced by a process-lifetime OS-backed advisory lock or equivalent | It prevents an independently launched second process from creating another GPU lane against the same deployment runtime. |
| GPU execution concurrency | **1** | It prevents simultaneous Qwen/Whisper/G1 workloads on the shared GPU. |
| Queue | Bounded, in memory; default maximum three waiting jobs | It provides explicit backpressure without premature distributed infrastructure. |
| Job identifier | `job_` plus 32 lowercase random hexadecimal characters | It is opaque, non-sequential, 128-bit, and satisfies the runtime safe-session syntax. |
| Upload root | `<ProductionRuntimeConfig.cache_root>/web-runtime` | It remains inside an allowed configured runtime root and outside every dataset. |
| Production storage roots | Dedicated deployment `cache_root` and `output_root`, never shared with scientific experiment, Validation/Test, or historical audit runs | It isolates Task06-owned derived artifacts so deployment retention cannot affect scientific records. |
| Baseline durability | In-memory job/result metadata; no restart recovery | It states the actual single-process guarantee and avoids false persistence claims. |
| Public result | Embed the Task06 success outcome serialization | It preserves public-safe scientific values without reconstructing them. |
| Public failure | A web-safe mapper omits exception type and all internal detail | It preserves the Task06 failure/result distinction with a smaller HTTP surface. |
| Public deployment | DICC compute + Cloudflare presentation/network/security | Cloudflare is not model compute, and no always-on rented GPU is required. |

The default queue depth is an operational setting, not a scientific setting.
`WEB_MAX_QUEUED_JOBS` may lower or raise the number of waiting jobs after an
operator reviews expected wait time and available disk space. GPU execution
concurrency remains fixed at 1 for the Task07 baseline; increasing it requires
a new architecture decision and GPU capacity validation.

GPU concurrency=1 means one execution lane for the one authorized Task07
server instance. The process singleton contract is part of that guarantee: a
second process using the same `WEB_RUNTIME_ROOT` must not start its own queue,
runtime graph, or worker.

### Future optional extensions

- SSE at `GET /api/v1/jobs/{job_id}/events`, with polling still fully supported;
- `DELETE /api/v1/jobs/{job_id}` for controlled cancellation/cleanup;
- Cloudflare Access, rate limiting, Turnstile, or stronger API protection;
- encrypted, public-safe completed-result snapshots and restart rehydration;
- external durable queues or object storage only after a real scale requirement;
- a separate authenticated job-history experience; and
- an optional developer-only Gradio surface that consumes, rather than
  bypasses, the same web/API contract.

None of these extensions is required by the baseline and none may change the
Frozen G1 scientific contract.

## 3. Frozen Task06 integration boundary

The future web worker has exactly one inference dependency:

```text
FastAPI route
  -> bounded JobManager
  -> single JobWorker
  -> ProductionExecutionService.execute(session_id, claim, video_path)
  -> ProductionRuntime.run(...)
  -> existing real service graph
  -> ProductionResultBuilder (already owned by Task06)
  -> ProductionExecutionOutcome
```

The web layer generates the safe `session_id`, preserves the exact focal claim,
and passes the server-controlled upload path. It consumes the returned
`ProductionExecutionOutcome`; it never constructs a scientific result.

The production CLI remains a separate shell entry point. FastAPI must not run
`python -m app.production_cli` or parse CLI stdout. The web layer must not
import or call closed sub-runners to perform inference.

### Frozen scientific invariants

No Web setting, request field, query parameter, environment variable, or UI
control may modify or reinterpret:

- backbone `microsoft/deberta-v3-base`;
- at most 24 evaluated Frozen G1-eligible units;
- maximum sequence length 256;
- class-wise maximum pooling across every evaluated eligible unit;
- `fake=0` and `real=1`;
- Top-5 explanation-only selection; or
- the rule that selection scores do not define the sample prediction pool.

Text, transcript, and OCR units may be Frozen-G1-eligible. Visual observations
remain supplemental with `eligible_for_frozen_g1=False`, no Frozen G1 logits,
no Frozen G1 selection score, and no confidence. The UI and API must never imply
otherwise.

The visual branch has two distinct claim semantics:

```text
focal claim + candidate frames
  -> SigLIP claim-conditioned / claim-relevance retrieval
  -> selected frames only (no focal claim passed onward)
  -> claim-blind Qwen observation generation
  -> supplemental visual RuntimeUnits
```

The entire visual pipeline is therefore not claim-blind. SigLIP selects frames
using claim relevance; Qwen generates observation text from those selected
frames without receiving the focal claim. The resulting RuntimeUnits remain
supplemental, `eligible_for_frozen_g1=False`, not scored by Frozen G1, and carry
no Frozen G1 logits, selection score, or confidence.

## 4. System and component topology

```text
Browser
  |
  | HTTPS: static assets
  v
Cloudflare Pages
  |
  | HTTPS: /api/v1 (short upload/status/result requests)
  v
Cloudflare network controls + optional future Access
  |
  | Cloudflare Tunnel when permitted
  v
DICC FastAPI process (one production process)
  |-- process-lifetime singleton server lock
  |-- health/readiness routes
  |-- upload validator and admission controller
  |-- bounded in-memory JobManager
  |-- one worker thread / execution lane
  |-- one long-lived ProductionExecutionService
  |-- one long-lived ProductionRuntime / real service graph
  `-- dedicated <cache_root>/web-runtime workspaces
          |
          `-- DICC A100 execution
```

Cloudflare serves and protects transport/presentation. It does not host models,
run inference, own the queue, or persist scientific results. The API origin is
the DICC/UM environment.

### Backend component responsibilities

| Component | Owns | Must not own |
| --- | --- | --- |
| Singleton guard | process-lifetime ownership of one validated `WEB_RUNTIME_ROOT` | PID-file-only locking, public process details, or distributed locking |
| FastAPI application | HTTP parsing, response mapping, lifespan | Scientific inference or result reconstruction |
| Upload validator | size, filename syntax, extension/MIME/container policy | Dataset discovery or arbitrary filesystem paths |
| Admission controller | queue-slot reservation and backpressure | GPU execution |
| `JobManager` | public state, timestamps, queue position, retention | Scientific stages or fake progress |
| `JobWorker` | serial call to `ProductionExecutionService.execute()` | Direct model/sub-runner calls |
| Task06 service | runtime execution and public-safe result/failure outcome | HTTP state or CORS |
| Cleanup service | web workspace and expired metadata cleanup | source data, model assets, runtime cache outside its root |

The synchronous production call must run outside the FastAPI event-loop thread.
The one worker owns the blocking call; health and polling requests remain
responsive while inference runs.

## 5. DICC and Cloudflare deployment topology

### Primary low-cost topology

```text
Cloudflare Pages: React/Vite static assets
        |
        | browser calls configured HTTPS API hostname
        v
Cloudflare proxy / future Access / rate controls
        |
        | outbound-established Cloudflare Tunnel
        v
DICC FastAPI on loopback or a restricted interface
        |
        v
ProductionExecutionService -> DICC A100
```

The frontend build receives its API base URL through a public build-time value
such as `VITE_API_BASE_URL`; no DICC hostname, path, credential, or tunnel token
is compiled into the browser bundle. API responses use `Cache-Control: no-store`.
Cloudflare caching must be bypassed for `/api/v1/*`.

The default demonstration upload limit is **90 MiB (94,371,840 bytes)**,
configurable downward. This leaves margin below Cloudflare's documented 100 MB
Free/Pro request-body ceiling. The operator must recheck the active plan and
zone-level limit before deployment; a larger backend limit does not override an
edge limit.

### Cloudflare security role

Cloudflare Tunnel is preferred over opening a DICC inbound port because the
connector establishes outbound connectivity. Future Cloudflare Access may
provide identity-aware admission in front of the API. Access is not assumed to
be configured in the initial baseline, and an opaque job ID is not a substitute
for authentication. Institutional DICC/UM policy remains authoritative.

Current operational references to revalidate during deployment:

- [Cloudflare Tunnel](https://developers.cloudflare.com/tunnel/)
- [Cloudflare Pages React deployments](https://developers.cloudflare.com/pages/framework-guides/deploy-a-react-site/)
- [Cloudflare request body limits](https://developers.cloudflare.com/workers/platform/limits/#request-limits)
- [Cloudflare Access for self-hosted applications](https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/self-hosted-public-app/)

## 6. Fallback deployment modes

### Mode A: direct DICC connector

```text
Browser -> Cloudflare -> cloudflared on DICC -> 127.0.0.1:8000 FastAPI
```

Use when DICC permits `cloudflared` and outbound Cloudflare connectivity. Bind
FastAPI to loopback, map only the API hostname, and keep the origin otherwise
unroutable.

### Mode B: Mac relay when DICC cannot run `cloudflared`

```text
DICC 127.0.0.1:8000 FastAPI
  <- SSH local port forward initiated by the user's Mac
Mac 127.0.0.1:<relay-port>
  <- cloudflared on the Mac
Cloudflare API hostname
  <- public frontend
```

The Mac establishes an authenticated SSH port forward to DICC; `cloudflared`
maps the public API hostname to the Mac loopback relay. Neither process binds a
new public DICC port. This mode depends on the Mac, SSH session, DICC login, and
Cloudflare connector all remaining available.

### Mode C: offline/local thesis-defense fallback

```text
DICC 127.0.0.1:8000 FastAPI
  <- SSH port forward
presenter's Mac 127.0.0.1:<local-port>
  <- local static frontend / browser
```

Cloudflare is absent. The same `/api/v1` contract and static frontend are used
with a loopback API base URL. This is the required defense fallback when public
ingress is unavailable.

## 7. Long-running inference constraint

A real Task06 video has taken approximately four minutes. Therefore:

- `POST /api/v1/jobs` ends after upload validation, workspace creation, and
  queue admission; it does not wait for inference;
- success returns **HTTP 202 Accepted** with a job identifier and polling link;
- inference continues in the one worker after the HTTP request ends;
- status and result retrieval are short independent requests; and
- no proxy timeout is used as an inference lifecycle mechanism.

No endpoint may report invented stage percentages such as “ASR 37%” or “G1
95%.” Only accepted, queued, running, terminal state, actual queue position,
timestamps, and elapsed time owned by the server are public.

## 8. Public job state machine

The exact public states are:

```text
accepted -> queued -> running -> completed -> expired
                            `-> failed -----> expired
```

Allowed transitions are only:

| From | To | Meaning |
| --- | --- | --- |
| `accepted` | `queued` | Validation, upload, and capacity reservation succeeded; the job is placed in the bounded queue. |
| `queued` | `running` | The sole worker claimed the job. |
| `running` | `completed` | Task06 returned `ProductionExecutionOutcome.status == success`, including successful NEI. |
| `running` | `failed` | Task06 returned `ProductionExecutionOutcome.status == failure`, or the web worker suffered a public-safe operational failure. |
| `completed` | `expired` | Result retention elapsed and result data was removed. |
| `failed` | `expired` | Failure retention elapsed and public failure metadata was removed. |

`accepted` is normally brief but remains explicit for race-free admission and
observable HTTP 202 responses. A queue-full request does not create a job and
returns 429. Invalid uploads do not create jobs. Active jobs never expire. No
cancellation transition exists in the baseline.

An `expired` tombstone remains available for a short configurable grace period
(default 10 minutes), after which the ID becomes indistinguishable from an
unknown ID and returns 404. This avoids indefinite public metadata retention.

## 9. Frontend and backend state distinction

Backend states are server truth: `accepted`, `queued`, `running`, `completed`,
`failed`, and `expired`.

Frontend workflow states are:

```text
idle -> file_selected -> submitting
                         |-> accepted -> queued -> running -> completed
                         |                         `-------> failed
                         |-> queue_full
                         `-> submission_error
```

`idle`, `file_selected`, `submitting`, client validation errors, and network
errors are browser-local. `accepted`, `queued`, `running`, `completed`,
`failed`, and `expired` are rendered only from an API response. A polling
network error does not change the last known backend job state; the UI reports
connection loss separately and retries with backoff.

## 10. GPU concurrency and bounded queue policy

### Invariants

- GPU execution concurrency = **1**.
- Exactly one active Task07 API/runtime process may use a configured
  `WEB_RUNTIME_ROOT`.
- Startup must acquire an exclusive process-lifetime singleton server lock
  before orphan cleanup, `JobManager` creation, `ProductionRuntime.start()`, or
  worker startup.
- The server lock must use an OS-backed advisory lock or an equivalent
  lifecycle-safe exclusive mechanism. Mere PID-file existence is not a lock.
- A second process that cannot acquire the lock fails startup safely and never
  becomes ready.
- The lock is held for the full server lifetime and released on clean shutdown.
- Lock paths and process IDs are never public. Redis, a database, and
  distributed locking are not baseline dependencies.
- Exactly one job worker may call `ProductionExecutionService.execute()` at a
  time.
- Production deployment uses one FastAPI process; multiple Uvicorn/Gunicorn
  workers are forbidden because each would create another in-memory queue and
  model graph.
- Auto-reload is forbidden in production.
- The bounded queue counts waiting jobs; the running job is separate.
- The default maximum is three queued jobs; the operator may configure the
  bound but not unbound it.
- Admission reserves a queue slot before the upload is accepted, preventing two
  simultaneous uploads from overcommitting one remaining slot.

If capacity is exhausted, `POST /api/v1/jobs` returns 429 with the stable public
code `queue_full` and a `Retry-After` hint. It does not accept the file, create a
job, or launch another worker. Readiness reports `accepting_jobs=false` while
full.

Queue position is one-based and appears only when the manager can compute it
from the current in-memory queue. It is advisory and may change when an earlier
job fails or leaves the queue. The UI must not derive completion time or a
percentage from it.

These controls jointly protect effective GPU concurrency=1. A single worker
serializes jobs inside the authorized process; the singleton lock prevents an
accidental second process using the same deployment runtime from introducing a
second execution lane.

## 11. Runtime and service lifecycle

On FastAPI lifespan startup, the baseline must:

1. load and validate web operational configuration;
2. verify that the dedicated web runtime root is safe and inside the configured
   production `cache_root`;
3. acquire the exclusive process-lifetime singleton server lock at a
   containment-safe location beneath the validated `WEB_RUNTIME_ROOT`;
4. only after lock acquisition, remove eligible orphan Task07 web workspaces;
5. construct exactly one `ProductionExecutionService` from the production
   runtime configuration;
6. call the owned runtime's idempotent `start()` once to run Task06 filesystem
   preflight and build the real service graph;
7. create the bounded manager and one worker; and
8. mark readiness true only after all startup validation succeeds.

Failure to acquire the singleton lock is a startup failure: no cleanup,
`JobManager`, Task06 runtime startup, or worker startup may occur, and readiness
must never become true for that process. The lock is not implemented as an
unverified PID file, and its path and owner process ID are not exposed through
health, readiness, errors, or logs intended for public consumption.

Task06 `ProductionRuntime.start()` performs preflight and graph construction;
it does not run inference and the inspected production services remain lazy
about model loading. The first real job may therefore still incur model-load
latency. The exact `ProductionExecutionService` and its runtime are reused by
every job. A runtime object is not recreated per request or per job.

On shutdown, the server stops admission, reports not ready, allows a bounded
graceful window for the running worker, stops the manager/runtime lifecycle,
then releases the singleton lock and exits. The lock remains held throughout
the shutdown sequence. Because the baseline has no durable queue, process
termination may lose nonterminal jobs. The API must not claim otherwise.

## 12. Upload, session, and workspace security

### Request fields

`POST /api/v1/jobs` uses `multipart/form-data` with exactly:

- `claim`: required string, nonblank after whitespace inspection, at most 2,000
  Unicode code points; the original accepted string is passed unchanged to
  Task06; and
- `video`: required single file.

Unknown multipart fields are rejected. Scientific/model parameters are not
accepted.

### Job and path construction

- Generate `job_<32 lowercase random hex characters>` with a cryptographic
  random generator; never accept a job/session ID from the caller.
- Use the same generated value as the Task06 `session_id`.
- `WEB_RUNTIME_ROOT` resolves to `<cache_root>/web-runtime` by default and must
  remain a descendant of the resolved configured `cache_root`.
- The configured root and job directories must not be symlinks. Each job uses
  only `<WEB_RUNTIME_ROOT>/jobs/<job_id>/`.
- Store the upload under a server-generated fixed basename such as
  `input.<validated-extension>`; never use the caller filename as a path.
- Resolve and containment-check every server-created path before writing, and
  use no-follow/exclusive creation semantics where the platform permits.
- Never traverse, read, or write UniRumor source dataset roots or their
  symlinks. Source datasets remain read-only and outside the web root.

The caller filename may be inspected only for policy. Reject NUL, `/`, `\`,
absolute paths, drive prefixes, or `..` path components. Do not persist or echo
the caller filename in public API responses or server correlation logs.

### Video policy

The initial allowlist is:

| Extension | Accepted MIME |
| --- | --- |
| `.mp4` | `video/mp4` |
| `.m4v` | `video/x-m4v`, `video/mp4` |
| `.mov` | `video/quicktime` |
| `.webm` | `video/webm` |

Extension and MIME must form an allowed pair, and a bounded server-side
container-signature probe must agree before enqueue. MIME and extension are
hints, not trust anchors. Generic `application/octet-stream` is rejected in the
baseline. A valid container check does not authorize files outside the web
workspace.

The maximum upload size is configurable, defaults to 90 MiB, and is enforced
both from `Content-Length` when present and while streaming. Reject an empty
upload. Abort, close, and delete any partial file when the stream exceeds the
limit or validation fails. File permissions must be owner-only on a shared
host.

### Cleanup lifecycle

- Submission failure: delete the partial upload and release the reserved slot.
- Queued/running: retain only the job workspace required by Task06.
- Completed or failed: delete the uploaded video and other web-owned temporary
  files immediately after Task06 returns and the public state is safely stored
  in memory.
- Terminal metadata: retain for a configurable period, default 60 minutes.
- Expiration: discard result/failure data, expose a 10-minute `expired`
  tombstone, then remove the job record.
- Startup: remove orphan job workspaces from the previous process because the
  baseline does not recover their jobs. This occurs only after the singleton
  lock is acquired. Cleanup must never recurse outside the validated web root.

### Storage ownership and Task06-derived artifacts

Task07 storage has two separate ownership classes:

**A. Task07-owned upload/workspace artifacts** live only under
`<cache_root>/web-runtime/jobs/<job_id>/`. Task07 owns their exact names and
lifecycle, so the upload cleanup rules above apply after containment and
symlink checks.

**B. Task06-owned derived artifacts** are created by the closed production
runtime in namespaces including:

```text
<ProductionRuntimeConfig.cache_root>/ocr
<ProductionRuntimeConfig.cache_root>/visual
<ProductionRuntimeConfig.cache_root>/g1
<ProductionRuntimeConfig.output_root>/g1
```

These may contain sampled OCR frames, candidate/selected visual frames, Frozen
G1 request and prediction records, and other per-session engineering artifacts.
They are not Task07 upload-workspace files.

The frozen retention boundary is:

1. Public web deployment uses dedicated production `cache_root` and
   `output_root` locations that are not shared with scientific experiments,
   Validation/Test, or historical audit runs.
2. Task07 cleanup must never recursively or blindly delete Task06 cache/output
   roots or the `ocr`, `visual`, or `g1` namespaces.
3. Future per-job deletion of Task06-derived artifacts requires an explicit
   ownership map to the opaque session/job ID, resolved-path containment proof,
   and tests demonstrating that it deletes only deployment-owned artifacts
   while preserving model assets, unrelated jobs, datasets, and symlinks.
4. Until ownership-aware per-job cleanup is implemented and tested, Task06-
   derived artifacts are retained inside the dedicated web deployment
   `cache_root`/`output_root` and controlled by a bounded operator/deployment
   retention policy. Unsafe broad deletion is forbidden.
5. Disk growth and retention of Task06-derived frames, request records, and
   prediction records are explicit Task07D and Task07H deployment gates.
6. No public API, health/readiness response, error, UI field, or public log may
   expose these artifacts or their local paths.

This contract changes no Task06 writer or artifact behavior. It only constrains
Task07 deployment isolation, retention, and future cleanup ownership.

## 13. Server restart and durability behavior

The baseline source of truth is process memory. Uploaded temporary video exists
on disk only to let the live worker execute. Completed results are retained in
memory for the terminal retention interval; persistent result snapshotting is
off in the baseline.

After a server restart:

- `accepted`, `queued`, and `running` jobs are invalidated;
- completed/failed jobs from the previous process are not rehydrated;
- their old URLs return 404 after startup orphan cleanup;
- abandoned web-owned files are deleted from the validated web runtime root;
- Task06-owned derived artifacts remain under the dedicated deployment
  `cache_root`/`output_root` until an ownership-aware retention mechanism safely
  removes them;
- no inference is resumed automatically; and
- the user must resubmit the original claim and file.

Future public-safe snapshots may add completed-result recovery, but the feature
must atomically store only the already public-safe Task06 serialization, define
an index and retention strategy, and never persist internal warnings or paths.

## 14. API conventions

All responses are JSON except the multipart request. Timestamps are RFC 3339
UTC strings. Durations are nonnegative integer milliseconds. Unknown fields may
be added in backward-compatible revisions, but frozen field meanings must not
change within `/api/v1`.

Every response carries a public correlation header such as `X-Request-ID`.
Errors use this envelope:

```json
{
  "api_version": "v1",
  "error": {
    "code": "stable_machine_code",
    "message": "Public-safe explanation.",
    "request_id": "req_opaque"
  }
}
```

The envelope never includes exception text, traceback, subprocess stderr,
internal warnings, or filesystem/model/cache/dataset paths.

### Public job resource

```json
{
  "api_version": "v1",
  "job": {
    "job_id": "job_0123456789abcdef0123456789abcdef",
    "state": "queued",
    "queue_position": 1,
    "created_at": "2026-08-20T10:00:00Z",
    "started_at": null,
    "finished_at": null,
    "expires_at": null,
    "queue_elapsed_ms": 2500,
    "execution_elapsed_ms": 0,
    "failure": null,
    "links": {
      "self": "/api/v1/jobs/job_0123456789abcdef0123456789abcdef",
      "result": "/api/v1/jobs/job_0123456789abcdef0123456789abcdef/result"
    },
    "poll_after_ms": 3000
  }
}
```

`queue_position` is null outside `queued`. `started_at` is null and
`execution_elapsed_ms` is zero until execution begins. `finished_at` is set only
for `completed`/`failed`. `expires_at` is set only for terminal jobs. `failure`
is non-null only for `failed` and contains only `code`, `message`, and an opaque
`incident_id`.

## 15. API endpoint contract

### `GET /api/v1/health`

Purpose: prove the HTTP process and event loop are alive.

- Request body: none.
- `200 OK`: `{"api_version":"v1","status":"ok"}`.
- Must not inspect model paths, run preflight, load models, start the runtime, or
  run inference.

### `GET /api/v1/readiness`

Purpose: state whether startup validation completed and new jobs can be
accepted.

Example ready response:

```json
{
  "api_version": "v1",
  "status": "ready",
  "accepting_jobs": true,
  "capacity_state": "available"
}
```

- `200 OK`: startup/preflight succeeded, the worker is alive, shutdown has not
  begun, and a queue reservation is currently available.
- `503 Service Unavailable`: startup/preflight failed, worker is unavailable,
  shutdown is in progress, or the queue is full. Include `Retry-After` when the
  condition is plausibly transient.
- Must not expose queue contents, model names/paths, hostnames, GPU identifiers,
  secrets, exceptions, or run inference.

### `POST /api/v1/jobs`

Purpose: validate one exact focal claim and one video, reserve capacity, store
the upload safely, create a job, and enqueue it without waiting for inference.

Request: `multipart/form-data` with `claim` and `video` only.

Success: `202 Accepted`, `Location: /api/v1/jobs/{job_id}`, and the public job
resource. The initial body may show `accepted`; the manager then transitions it
to `queued`.

Relevant errors:

| Status | Stable code | Condition |
| --- | --- | --- |
| `400` | `malformed_request` | Invalid multipart syntax or duplicate required part |
| `413` | `upload_too_large` | Declared or streamed body exceeds the configured limit |
| `415` | `unsupported_video_type` | Extension/MIME/container policy fails |
| `422` | `invalid_claim` | Claim is absent, blank, or too long |
| `422` | `empty_upload` | Video has zero bytes |
| `422` | `invalid_filename` | Filename contains forbidden path syntax |
| `429` | `queue_full` | No bounded-queue reservation is available |
| `503` | `service_not_ready` | Startup, worker, or shutdown state forbids admission |

Rejected requests create no public job. Queue-full and not-ready responses may
include `Retry-After`; they must not reveal other jobs.

### `GET /api/v1/jobs/{job_id}`

Purpose: return current server-owned job state and real timing.

- `200 OK`: `accepted`, `queued`, `running`, `completed`, or `failed` job
  resource.
- `410 Gone`: known `expired` tombstone with code `job_expired`.
- `404 Not Found`: malformed/unknown ID, using one indistinguishable response.

The endpoint never returns model-stage percentages. It may return queue
position only when exactly known.

### `GET /api/v1/jobs/{job_id}/result`

Purpose: return a scientific result only after successful completion.

Completed response:

```json
{
  "api_version": "v1",
  "job_id": "job_0123456789abcdef0123456789abcdef",
  "outcome": {
    "schema_version": 1,
    "status": "success",
    "result": {
      "schema_version": 1,
      "session_id": "job_0123456789abcdef0123456789abcdef",
      "claim": "Exact focal claim",
      "verdict": {
        "model_verdict": "fake",
        "display_verdict": "Fake",
        "evidence_status": "sufficient",
        "sample_logits": {"fake": 1.25, "real": -0.25},
        "probabilities": {"fake": 0.82, "real": 0.18},
        "class_winners": {"fake": "transcript-001", "real": "transcript-001"},
        "checkpoint_sha256": "public-checkpoint-digest"
      },
      "sufficiency": {
        "status": "sufficient",
        "reason_code": "frozen_g1_evidence_available_and_model_completed",
        "model_was_run": true,
        "g1_exposure_count": 1,
        "transcript_exposure_count": 1,
        "ocr_exposure_count": 0,
        "visual_unit_count": 0,
        "top_k_count": 1,
        "supplemental_visual_present": false
      },
      "evidence": {
        "g1_exposure_units": [
          {
            "unit_id": "transcript-001",
            "source_type": "transcript",
            "text": "Public-safe transcript evidence",
            "start_time": 1.0,
            "end_time": 3.0,
            "frame_id": null,
            "bbox": null,
            "confidence": null,
            "producer": "public-producer-id",
            "eligible_for_frozen_g1": true,
            "selection_score": 0.8,
            "logits": {"fake": 1.25, "real": -0.25},
            "extraction_method": "public-extraction-method",
            "source_index": 0,
            "frame_ids": [],
            "evidence_refs": [],
            "source_unit_ids": [],
            "observation_type": null
          }
        ],
        "g1_top_k_explanation_unit_ids": ["transcript-001"],
        "visual_supplemental_units": []
      },
      "runtime_ms": 240000.0
    },
    "failure": null
  }
}
```

The real `outcome` is the existing Task06
`ProductionExecutionOutcome.to_dict()` success representation. It is embedded,
not rebuilt or numerically transformed.

- `200 OK`: `completed`, including Fake, Real, and successful NEI.
- `409 Conflict` with `job_not_completed`: accepted/queued/running.
- `409 Conflict` with `job_failed`: operational failure; no result or verdict.
- `410 Gone`: expired.
- `404 Not Found`: malformed/unknown ID.

### Reserved endpoints, not implemented in the baseline

- `DELETE /api/v1/jobs/{job_id}`: future cancellation and cleanup semantics.
- `GET /api/v1/jobs/{job_id}/events`: future SSE state events.

No WebSocket endpoint is required.

## 16. Health versus readiness

Health is liveness: the HTTP process can answer. It stays cheap, returns 200
while the process is alive, and performs no runtime action.

Readiness is admission capability: configuration and Task06 filesystem
preflight passed, the singleton server lock is held, the service graph was
built, the single worker is alive, the server is not draining, and the bounded
queue can reserve a slot. It may return 503 while health remains 200. A second
process that failed lock acquisition never becomes ready. Neither endpoint
loads a model or runs inference.

## 17. Public failure and NEI semantics

The mapping is exact:

| Task06 outcome | Job state | Public meaning |
| --- | --- | --- |
| `status == success`, Fake | `completed` | Successful binary model result |
| `status == success`, Real | `completed` | Successful binary model result |
| `status == success`, NOT_RUN/NEI | `completed` | Successful engineering abstention due to insufficient eligible evidence |
| `status == failure` | `failed` | Operational execution or result-packaging failure |

NEI means “Insufficient eligible evidence for a Frozen G1 decision.” It is not
a learned class, probability threshold, server error, or failed job.

Operational failure must never become Fake, Real, or NEI. The job status mapper
may expose the stable Task06 failure code and fixed public message, but it omits
`exception_type` from the browser contract. Raw exception messages, tracebacks,
causes, subprocess stderr, DICC paths, model/cache/dataset paths, and internal
warnings are server-side only.

## 18. Result contract usage

The completed response preserves Task06 fields and their meaning:

- verdict: model/display verdict, evidence status, exact sample logits,
  probabilities, class winners, and checkpoint digest;
- structural sufficiency assessment;
- ordered full G1 exposure units;
- ordered `g1_top_k_explanation_unit_ids`;
- separately ordered supplemental visual units; and
- runtime in milliseconds.

The frontend may format numbers for display but must retain access to the exact
serialized values in technical details. It must not recompute probabilities or
derive a different verdict.

Top-5 is **explanation-only** and not the only prediction basis. The binary
sample result is based on class-wise maximum pooling over every evaluated
eligible unit. Supplemental visual evidence is separately labeled, remains
Frozen-G1-ineligible, and has no Frozen G1 logits, selection score, or
confidence.

## 19. CORS and network configuration

Baseline public deployment uses separate HTTPS origins, for example a Pages
frontend and `api.<controlled-domain>`.

- Configure the frontend API base URL per build/deployment; never infer it from
  a DICC path or hard-code a tunnel URL in source.
- FastAPI allows only exact configured frontend origins plus explicit local
  development origins. Do not use `*`.
- Baseline requests use no browser credentials/cookies. Allow only required
  methods (`GET`, `POST`, `OPTIONS`) and required headers (`Content-Type` and
  correlation-related response headers).
- If Cloudflare Access later introduces credentials, re-evaluate cookies,
  preflight, token validation, Pages preview domains, and origin allowlists as
  one security change.
- Public API transport is HTTPS. Mode C uses loopback HTTP only inside the SSH
  forwarding boundary.
- Apply `Cache-Control: no-store` to job/status/result responses and configure
  Cloudflare not to cache `/api/v1/*`.

Polling baseline:

1. honor server `poll_after_ms` or `Retry-After`;
2. poll every approximately 2 seconds for `accepted` and 3 seconds for queued or
   running, with about 10% jitter;
3. on consecutive network failures, back off exponentially to a 30-second cap;
4. while the tab is hidden, reduce polling to approximately every 15 seconds and
   refresh immediately when visible; and
5. stop on completed, failed, expired, or unknown job.

Polling remains complete even if SSE is added later.

## 20. Privacy, redaction, and observability

Server logs and public responses are separate contracts. Structured server
logging should use a request ID and opaque job ID for correlation and record
state changes, coarse durations, queue admission, cleanup, and Task06 failure
code. It must not make public logs available through an API route.

Public responses and frontend telemetry must not contain:

- caller or server absolute filenames;
- upload workspace paths;
- DICC host paths or source dataset paths;
- local model/config/cache/output paths;
- raw Python exception text or traceback;
- subprocess stderr;
- internal warnings; or
- claim/video content in URLs or analytics.

The same prohibition covers Task06-owned sampled frames, selected/candidate
visual frames, Frozen G1 request/prediction records, and every derived artifact
path under the deployment `cache_root` or `output_root`.

There is no job-list endpoint. Job IDs are non-enumerable capability-like
references, but they are not authentication. Set a restrictive referrer policy
and avoid third-party telemetry in the baseline.

## 21. Low-cost deployment objective

The primary cost model is:

- DICC supplies existing A100 model compute;
- Cloudflare Pages supplies static frontend hosting where practical;
- Cloudflare supplies network/security controls and Tunnel ingress when allowed;
- local development serves the frontend locally; and
- SSH forwarding supplies a thesis-defense fallback.

There is no always-on rented cloud GPU baseline. External availability depends
on DICC network reachability, user/Mac relay availability in Mode B, and UM/DICC
institutional policy. This contract does not claim that DICC has authorized
public hosting; authorization is an unresolved deployment gate.

## 22. Frontend architecture and conceptual directory structure

The final primary UI uses React, TypeScript, Vite, Tailwind CSS, Framer Motion,
Radix primitives with shadcn-compatible patterns, and Lucide icons. It builds to
static assets and owns no server secrets.

Conceptual Task07B structure (not created by Task07A-1):

```text
frontend/
  src/
    api/          # /api/v1 client, envelopes, polling transport
    components/   # design-system primitives and shared compositions
    features/
      verification/
      jobs/
      results/
      evidence/
    hooks/        # job polling, reduced motion, theme
    layouts/      # application shell and result workspace
    pages/        # verification and job routes
    styles/       # semantic tokens and global foundations
    types/        # public API/result types only
```

The frontend state model is discriminated by explicit workflow/job state. It
must not infer backend progress from timers or animation.

## 23. Task07 implementation sequence

1. **Backend contract types and configuration.** Add FastAPI/web configuration,
   request/response types, safe path helpers, and contract tests without model
   execution.
2. **Singleton, job manager, and worker.** Implement exclusive process-lifetime
   server locking, lock-contention startup failure, exact transitions, bounded
   reservation, effective concurrency=1, retention, restart cleanup, and
   deterministic fake-service tests. Tests must prove no cleanup/runtime/worker
   startup occurs before lock acquisition and that a second process never
   becomes ready.
3. **API routes.** Implement `/api/v1`, upload streaming/validation, public
   outcome mapping, CORS, health/readiness, and failure-redaction tests.
4. **Frontend foundation.** Scaffold the independent React/TypeScript/Vite app,
   semantic design tokens, state reducer, typed API client, and polling.
5. **Enterprise workflow.** Build upload, queue/running, result/evidence,
   technical disclosure, both themes, responsive layouts, and accessibility.
6. **Integration verification.** Exercise frontend/backend with a stub service,
   then one authorized DICC real run without Validation/Test access.
7. **Deployment.** Validate Mode C first, then permitted Tunnel mode, exact CORS,
   edge upload limits, Access/rate policy, singleton-lock behavior, and public
   path redaction. At Task07D and again at Task07H, measure Task06-derived
   artifact disk growth, set a bounded operator retention policy, and verify
   dedicated deployment `cache_root`/`output_root` isolation before release.

Each phase must continue to treat Task06 as closed production code.

## 24. Explicit unresolved deployment questions

These questions must be answered by the operator or institution before public
deployment; they do not block local Task07 implementation:

1. Does UM/DICC policy authorize a public or Access-protected API tunnel from
   the compute environment?
2. May `cloudflared` run on the DICC node, or is Mode B's Mac relay required?
3. Which controlled domain and exact frontend/API origins will be used?
4. Which Cloudflare plan and zone-level request-body limit apply on deployment
   day?
5. Is Cloudflare Access required at launch, and which identities/policies are
   approved?
6. What request/rate limits and public demonstration audience are authorized?
7. What DICC disk quota is available for the web runtime root and queue depth?
8. What terminal-result retention period is permitted for claims and evidence?
9. What service start/stop and SSH/Tunnel supervision mechanism is allowed on
   the host?
10. Is the DICC node available during the thesis defense, and has Mode C been
    rehearsed on the presentation network?
11. Does the selected `WEB_RUNTIME_ROOT` filesystem provide reliable
    process-lifetime advisory locking for the singleton server contract?

## 25. Architecture acceptance gates

Task07 implementation is conformant only if all of the following are true:

- the web worker calls only `ProductionExecutionService` for inference;
- the production CLI remains separate;
- `/api/v1` uses HTTP 202 plus polling for long-running jobs;
- GPU concurrency = 1 and the queue is bounded with explicit 429 backpressure;
- exactly one active Task07 process per `WEB_RUNTIME_ROOT` holds the singleton
  server lock for its complete lifetime, and lock contention fails startup;
- one service/runtime graph is reused by the one authorized server process;
- upload paths are server-generated beneath the validated web runtime root;
- production `cache_root` and `output_root` are dedicated to this web
  deployment and are not shared with scientific or historical runs;
- Task07 cleanup never broadly deletes Task06-owned derived artifacts; any
  future deletion is ownership-mapped and containment-proven;
- Task07D/Task07H gate Task06-derived artifact disk growth and retention;
- source datasets and Validation/Test data are never read or written;
- restart limitations and retention are visible and honest;
- successful NEI completes while operational failure fails;
- Task06 success serialization is preserved without scientific recomputation;
- Top-5 explanation-only and supplemental visual evidence labels are explicit;
- no public response exposes internal paths, logs, or exceptions;
- React/TypeScript remains independently deployable as static assets;
- Cloudflare remains a presentation/network/security layer, not model compute;
- no rented cloud GPU is required by the baseline; and
- the design-system, reduced-motion, responsive, and accessibility contracts in
  `ENTERPRISE_UI_DESIGN_SYSTEM.md` are satisfied.
