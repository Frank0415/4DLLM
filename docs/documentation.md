# Workbench and analysis workflow

## Launch and configuration

Run `./run.sh` for the local browser interface or `uv run --locked python main.py` for MCP. Node.js/npm is needed for the initial frontend build. The application binds to `127.0.0.1`.

Settings are stored in `config/database.json` and `config/api_keys.json`. The browser can update them without returning existing passwords or API keys. Leaving secret fields blank preserves their values. `DATABASE_URI` overrides database settings; `FOURDLLM_CONFIG` chooses a database configuration file. `FOURDLLM_DATA_DIR` chooses the data directory, defaulting to `workspace/`.

Create the PostgreSQL database before connecting. The application then runs `SQL/schema.sql` to create missing tables, columns, indexes, and constraints without dropping existing data. To perform this step manually:

```bash
uv run --locked python setup_database.py --config config/database.json
```

## Import

The browser lists configured local directories through the backend. Large files are read directly by Python instead of being uploaded through the browser. Single-file import inspects headers and frame counts, copies MIB and HDR into a private staging directory, computes SHA-256, converts actual scan rows to MAT files, and writes scan references in one database transaction. Diffraction points use PostgreSQL COPY.

Duplicate scan names and duplicate stored SHA-256 values are rejected. Old scans without a digest cannot be recognized as content duplicates. Failed or cancelled imports roll back catalog writes and remove this import's staged files.

Supported formats are `U08`, `U16`, and `U32`, with detector layouts `1x1`, `2x2`, and `2x2G`. Packed `R64` is unsupported. Explicit dimensions override metadata; rows × columns must equal the frame count. Missing dimensions require user input, except that 65536 frames retain the original 256×256 default. Keep the companion `.hdr` beside the `.mib`.

Directory import processes MIB files immediately inside the selected directory, without recursion. Optional dimensions apply to the entire batch.

## Analysis and history

The analysis core loads one MAT row at a time and extracts features in batches. It retains the feature matrix for PCA and K-Means instead of concatenating the full raw scan into float32. Feature memory still grows with the number of patterns; PCA is not an unlimited streaming implementation.

Each clustering run receives a separate result directory:

| File | Contents |
|---|---|
| `clustering.npz` | `labels`, `xs`, `ys`, distances to cluster centers; coordinates are one-based |
| `parameters.json` | Seed, device selection, polar feature settings, PCA and K-Means settings |
| `cluster_map.png` | Neutral cluster map |
| `montages/cluster_N.png` | Nearest representative patterns for each cluster |
| `individual_images/cluster_N/` | Individual representative images |

`diffraction_patterns` stores current labels, while `pattern_cluster_assignments` retains the assignments of each run. The browser can select historical runs. GUI, MCP, and CLI clustering share a process lock.

## LLM and human annotation

LLM requests use Gemini's `generateContent` format, directly or through a compatible proxy. The default pipeline selects five evenly spaced samples per cluster and uses three concurrent requests.

Responses require a valid `classification_code`: 0 vacuum, 1 crystalline, 2 amorphous, 3 mixed. Failed or invalid responses remain errors. Cluster numbers never serve as fallback material labels. The model, prompt, and returned samples are retained with the analysis and task report.

Human category and note are saved separately in `cluster_annotations`. Repeated LLM analysis updates the displayed suggestion; previous browser task reports remain available. The browser records the target clustering run when submitting an LLM task.

## Tasks

Task JSON and logs are persisted under `workspace/jobs/`. A serial queue launches independent Python workers. Status values are `queued`, `running`, `succeeded`, `failed`, and `cancelled`.

Browser refresh or closure does not stop processing. Cancellation is cooperative, taking effect at processing checkpoints. PCA, K-Means, and current LLM network requests finish before cancellation is observed. Already committed operations are not undone; a task that has committed its final result reports success. LLM clusters commit separately, so earlier completed cluster results survive a later failure or cancellation.

Normal shutdown waits for the active worker. Queued tasks continue after restart, while crashed workers are marked failed. Retry creates a new task rather than resuming a partially completed step.

## HTTP endpoints

All paths below have the `/api` prefix.

| Method | Path | Purpose |
|---|---|---|
| GET / PUT | `/settings` | Read displayable settings / save configuration |
| GET | `/status`, `/files?path=...` | Connection status / directory listing |
| POST | `/inspect`, `/imports`, `/import-folders` | Inspect or import files; bodies include `path`, optional `scan_shape` |
| GET | `/scans`, `/scans/{id}` | Scans and run history |
| GET | `/scans/{id}/overview`, `/scans/{id}/pattern?row=1&col=1` | Preview images |
| POST | `/scans/{id}/cluster` | Submit `k`, `seed`, optional `device` |
| POST | `/scans/{id}/llm` | Submit optional `cluster_id`, `samples`, `concurrency` |
| GET | `/runs/{id}`, `/runs/{id}/grid`, `/runs/{id}/files/{name}` | Results, spatial labels, artifacts |
| PUT | `/clusters/{id}/annotation` | Save `category` and `note`; null category clears the annotation |
| GET | `/jobs`, `/jobs/{id}`, `/jobs/{id}/log`, `/jobs/{id}/report` | Task records, logs, downloadable reports |
| POST | `/jobs/{id}/cancel`, `/jobs/{id}/retry` | Request cancellation or retry |

The frontend lives in `frontend/`. For development, start the backend with `uv run --locked python -m workbench --no-browser`, and run `npm run dev` inside `frontend/`. Production frontend assets are built with `npm run build` and served by FastAPI.

## MCP and CIF

Scan import, clustering, image display, LLM, and CIF tools remain available. `ingest_scan_from_mib` accepts optional `scan_rows`, `scan_columns`, and `scan_name`. `check_analysis_status` returns browser task records. Synchronous MCP and CLI calls do not join the browser queue.

Disabled database tuning, health, Explain, and top-query modules were removed. SQL execution and read-only SQL validation remain.

Install optional CIF simulation dependencies using `uv sync --extra simulation`. CIF pattern comparison remains unimplemented and returns `not_implemented`. The first browser interface covers scan import, exploration, clustering, and LLM analysis.
