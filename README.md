# Workshop: Build a Durable Research Agent

In this workshop you'll build a small agentic workflow and see why **durable execution** matters the moment real money (tokens, API quotas, downstream side effects) is on the line.

The goal: an agent that takes a research topic and a target count, searches the web, and appends each matching entry to `leads.csv` — reliably, even through crashes and restarts.

## What you'll build

Three functions:

- `research_workflow(topic, target_count)` — the agent loop (DBOS workflow).
- `call_model(...)` — one turn against OpenAI's Responses API (DBOS step).
- `append_lead_to_csv(...)` — the side-effecting tool the model calls (DBOS step).

The model gets two tools: OpenAI's hosted `web_search` (resolved server-side, no extra API key) and your local `save_lead` function. The loop keeps calling the model until the CSV holds `target_count` rows.

## Prerequisites

- A running Postgres container: `docker run -d --name dbos-postgres -p 5432:5432 -e POSTGRES_PASSWORD=dbos pgvector/pgvector:pg16`
- An `OPENAI_API_KEY`

## Path A — Building with Claude Code

1. **Install the DBOS agent skills** (one-time setup):

   ```bash
   npx skills add dbos-inc/agent-skills
   ```

   This installs the `dbos-python` skill (along with the Go and TypeScript ones) under `~/.claude/skills/` so Claude Code can load it.

2. Start Claude Code in an empty directory.
3. Load the DBOS Python skill so Claude knows the framework's conventions:

   ```
   /skill dbos-python
   ```

4. Give Claude a prompt like:

   > Build a minimal DBOS Python app called `research-agent`. It should expose a single workflow `research_workflow(topic, target_count)` that runs an agent loop against OpenAI's Responses API. The model has two tools: the hosted `web_search` tool and a local `save_lead` tool implemented as a DBOS step that appends a row to `leads.csv`. Every LLM call must also be a DBOS step so turns are checkpointed. Use a deterministic workflow ID derived from the args so re-running with the same arguments resumes the same workflow. Keep it to a single `main.py` plus `pyproject.toml` and `dbos-config.yaml`.

5. Ask Claude to explain the durability guarantees (what happens if you kill the process mid-loop) before running it.

Tips:

- If Claude reaches for a separate web-search library, remind it to use OpenAI's built-in `{"type": "web_search"}` tool in the Responses API.
- If Claude wraps everything in one big function, push back: each external I/O (LLM call, CSV write) belongs in its own `@DBOS.step()`.

## Path B — Building manually

Read `main.py` as a reference. The skeleton:

```python
@DBOS.step()
def append_lead_to_csv(name, description, source_url):
    # Append one row to leads.csv. Checkpointed — never runs twice.

@DBOS.step()
def call_model(input_items, previous_response_id):
    # One turn of client.responses.create(...) with the two tools.
    # Return {"id": ..., "output": [serialized items]}.

@DBOS.workflow()
def research_workflow(topic, target_count):
    # Loop:
    #   1. call_model(...)
    #   2. For each function_call in the response, run append_lead_to_csv
    #      and collect a function_call_output.
    #   3. Exit when target_count is reached or the model stops calling tools.
    #   4. Otherwise feed tool outputs back via previous_response_id.
```

Two things that are easy to get wrong:

1. **LLM calls must be steps.** If the workflow calls the Responses API directly, DBOS cannot memoize the turn — a crash means re-calling (and re-paying for) every previous turn on recovery.
2. **Send only new tool outputs on subsequent turns.** `previous_response_id` threads the full conversation server-side. Resending the full history works but wastes tokens.

## Run it

Start the server — it stays up and accepts workflow requests over HTTP:

```bash
export OPENAI_API_KEY=...
export DBOS_SYSTEM_DATABASE_URL=postgresql://...

export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())") # fix ssl certs on some platforms
uv sync
uv run python main.py
```

### Optional: connect to DBOS Conductor

Set `DBOS_CONDUCTOR_KEY` before launch to stream workflow telemetry to [DBOS Conductor](https://console.dbos.dev) under the app name `workshop`:

```bash
export DBOS_CONDUCTOR_KEY=<your-conductor-key>
uv run python main.py
```

With the key set, you can watch workflows execute, inspect steps, and replay runs from the Conductor dashboard. The app runs fine without it.

Kick off a research run:

```bash
curl -X POST http://localhost:1234/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "Series A fintech startups in Europe", "target_count": 10}'
# → {"workflow_id": "research-series-a-fintech-...", "status": "started"}
```

Or enqueue it onto a DBOS queue (managed concurrency, useful for fan-out):

```bash
curl -X POST http://localhost:1234/research/enqueue \
  -H "Content-Type: application/json" \
  -d '{"topic": "Series A fintech startups in Europe", "target_count": 10}'
```

Check its status:

```bash
curl http://localhost:1234/research/<workflow_id>
```

List recent runs:

```bash
curl http://localhost:1234/research
```

## The durability demo

This is the point of the workshop — run it for yourself:

1. POST a research request. Watch leads appear in `leads.csv`.
2. Kill the server with **Ctrl+C** mid-run.
3. Restart the server: `uv run python main.py`.
4. Observe: DBOS auto-recovers the pending workflow on startup. Already-saved leads are not re-saved. Already-completed LLM turns are not re-called. The agent picks up from the first unfinished step and keeps going until it hits `target_count`.

Now imagine the same agent running 10,000 times a day against a paid search API, a flaky LLM provider, and a production database. Durable execution is the difference between "resilient by design" and "a pager that rings every deploy."
