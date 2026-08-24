# Production deployment

Four hosts, split along the line where memory actually hurts.

```
Browser
   |
   v
Vercel  ──  static Vite build (index/app/auth/feedback)
   |        rewrites /api/* and /health to Render, so the app talks
   |        same-origin and CORS never enters the picture
   v
Render  ──  FastAPI: auth, feedback, geocoding, distance matrix, job queue
   |        Dockerfile.render, ~469 MB image, ~70 MB resident
   |        no torch, no numba, no vrptw package
   v
Cloud Run ── solver: torch + numba + OR-Tools + the vrptw research package
   |         solver_service/Dockerfile, 2 vCPU / 4 GiB, scales to zero
   v
Hugging Face Hub ── DDQN-ALNS controller weights, pulled at container boot
```

Render's free web service has 512 MB of RAM. Importing torch alone exceeds that
before a single customer is routed, which is why the solver is a separate
service rather than a thread in the API process.

## Live resources

| Layer   | URL                                                                                     |
| ------- | --------------------------------------------------------------------------------------- |
| Web     | https://vrptw-research-optimization.vercel.app                                            |
| API     | https://vrptw-backend.onrender.com (Blueprint `vrptw-production`, branch `Simplyfile`)   |
| Solver  | https://vrptw-solver-700938471158.asia-southeast1.run.app (project `august-lamp-499804-h3`, `asia-southeast1`) |
| Weights | [`oggishi/vrptw-ddqn-alns`](https://huggingface.co/oggishi/vrptw-ddqn-alns)               |

A cold solve through the whole chain — Vercel rewrite, Render wake-up, Cloud Run
cold start, seven algorithms — takes roughly 40 s. Warm, it is the solver time
alone.

## How the split works

`src/backend/services/compute_gateway.py` is the seam. When `SOLVER_REMOTE_URL`
is set, these forward to the solver instead of running in-process:

| API route                    | Solver route      |
| ---------------------------- | ----------------- |
| `POST /api/jobs` (via worker) | `POST /solve`     |
| `POST /api/reoptimize`        | `POST /reoptimize` |
| `POST /api/solve/dynamic_insert` | `POST /dynamic_insert` |

`/api/benchmark`, `/api/train/*` and `/api/smoke-test` return **503** in this
configuration. They are multi-hour research jobs that write into `docs/logs`;
run them locally against the full stack, not against production.

Leave `SOLVER_REMOTE_URL` unset and every one of those runs in-process again —
that is the local development path, and it is what `Dockerfile` (the full image)
still does.

## Deploying

### 1. Weights → Hugging Face

```bash
python scripts/publish_model_hf.py --repo oggishi/vrptw-ddqn-alns
```

Needs `hf auth login` or `HF_TOKEN`. Re-run after any retraining; the solver
picks up the new checkpoint on its next cold start, no image rebuild.

### 2. Solver → Cloud Run

```bash
python scripts/deploy_solver.py --project august-lamp-499804-h3 \
    --hf-model-repo oggishi/vrptw-ddqn-alns
```

The script stages a minimal build context (~1.7 MB: `src/vrptw`, `src/backend`,
`data/Solomon`, the weights) rather than uploading the repository, generates a
`SOLVER_API_TOKEN` if you do not pass one, and prints the service URL and token
at the end. Both go into Render.

Requires `run.googleapis.com`, `cloudbuild.googleapis.com` and
`artifactregistry.googleapis.com` enabled on the project.

Sizing notes: `--concurrency 2` because a solve fans out over a
`ProcessPoolExecutor` and saturates every core it is given, so overlapping
requests on one instance only make each slower. `--min-instances 0` keeps idle
time free; `--cpu-boost` keeps the torch import off the first request's
critical path. Cold start is roughly 20-30 s.

### 3. API → Render

`render.yaml` is a Blueprint pointing at `Dockerfile.render`. Create the service
from it, then set the secrets it declares with `sync: false`:

| Variable              | Value                                                       |
| --------------------- | ----------------------------------------------------------- |
| `SOLVER_REMOTE_URL`   | Cloud Run service URL, no trailing slash                     |
| `SOLVER_API_TOKEN`    | The token from step 2                                        |
| `CORS_ALLOW_ORIGINS`  | Vercel production origin (only used for direct-origin calls) |
| `FRONTEND_URL`        | Vercel production origin, for password-reset links           |
| `DEMO_AUTH_BYPASS`    | `false` once Firebase credentials are in place               |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | Service account JSON, inline                     |

Health check is `/api/health`, which also reports whether the solver is
reachable:

```json
{"remote_solver": {"configured": true, "reachable": true, "detail": {...}}}
```

The free instance sleeps after 15 minutes idle; the first request afterwards
pays roughly 30 s of cold start on top of the solver's own.

### 4. Web → Vercel

`vercel.json` rewrites `/api/*` to the Render service, so `constants.js` keeps
resolving `API_BASE` to the origin-relative `/api` and no build-time API URL is
needed. Deploy with `vercel --prod` from the repository root.

`.vercelignore` keeps the backend, the research package and the datasets out of
the upload — Vercel only ever builds `src/frontend`.

One limit to know about: an external rewrite is a proxied request, so the
synchronous `/api/reoptimize` and `/api/solve/dynamic_insert` can hit the proxy
timeout if Render and Cloud Run are both cold at once. Solving goes through the
async job queue and is unaffected. If those two start timing out in practice,
the fix is `--min-instances 1` on Cloud Run, which trades the free tier for a
warm container.

## Known gap: the Render MCP server

`claude mcp add render https://mcp.render.com/mcp` installs, but the connection
fails with *"Incompatible auth server: does not support dynamic client
registration"* — Render's hosted MCP cannot complete the OAuth handshake from
this client. The Hugging Face MCP server connects fine.

To drive Render from an agent, add the server with an API key from
[Account Settings → API Keys](https://dashboard.render.com/u/settings#api-keys):

```bash
claude mcp add --scope user --transport http render https://mcp.render.com/mcp \
    --header "Authorization: Bearer rnd_..."
```

Render API keys are broadly scoped — they reach every workspace and service the
account can. The Blueprint flow in the dashboard needs no key at all.

## Verifying a deployment

```bash
curl -s https://<render-host>/api/health              # remote_solver.reachable == true
curl -s https://<cloud-run-host>/health               # weights_origin.source == "huggingface"
```

Then submit a job through the UI and confirm all seven algorithms come back:
`ddqn`, `alns`, `ortools`, `hybrid_fixed`, `hybrid_ddqn`,
`hybrid_ddqn_transfer_rc1`, `hybrid_ddqn_transfer_dr`.

## Cost

Cloud Run stays inside the always-free tier at demo traffic (2M requests,
180k vCPU-seconds, 360k GiB-seconds per month) because it scales to zero.
Render's web service and the Vercel hobby plan are free. Cloud Build is the one
metered piece: 120 build-minutes per day are free and a solver image takes
about 8 of them.
