"""Durable research agent — DBOS workshop example.

The workflow runs an agent loop against OpenAI's Responses API. The model has
access to two tools:
  1. `web_search` — hosted by OpenAI, resolved server-side.
  2. `save_lead`  — a local DBOS step that appends a row to `leads.csv`.

The loop keeps calling the model until it has saved `target_count` leads or the
model stops issuing tool calls. Every LLM call and every CSV append is a DBOS
step, so if the process dies mid-run, DBOS replays completed steps from their
checkpoints and resumes from the next unfinished one — no duplicate searches,
no duplicate CSV rows, no wasted tokens.
"""

import csv
import json
import os
from pathlib import Path

import uvicorn
from dbos import DBOS, DBOSConfig, Queue
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI()
research_queue = Queue("research_queue")

CSV_PATH = Path(__file__).parent / "leads.csv"
CSV_HEADERS = ["name", "description", "source_url"]
MAX_ITERATIONS = 20
MODEL = "gpt-4.1"

openai_client = OpenAI()

SYSTEM_PROMPT = """You are a research agent.

Given a topic and a target number of leads, find distinct matching entries and
save each one by calling the `save_lead` tool. Use `web_search` to discover
information from the live web. Save each qualifying entry as soon as you find
it — do not batch them. Stop once you have saved {target_count} leads."""

SAVE_LEAD_TOOL = {
    "type": "function",
    "name": "save_lead",
    "description": "Append a qualifying lead to the research CSV.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Entity name."},
            "description": {
                "type": "string",
                "description": "One or two sentences explaining why it matches the topic.",
            },
            "source_url": {
                "type": "string",
                "description": "URL of the primary source backing this entry.",
            },
        },
        "required": ["name", "description", "source_url"],
        "additionalProperties": False,
    },
}


@DBOS.step()
def append_lead_to_csv(name: str, description: str, source_url: str) -> str:
    """Side-effecting tool — appends one row to leads.csv. Memoized by DBOS."""
    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADERS)
        writer.writerow([name, description, source_url])
    DBOS.logger.info(f"Saved lead: {name}")
    return f"Saved {name}"


@DBOS.step()
def call_model(input_items: list, previous_response_id: str | None) -> dict:
    """Single turn against the Responses API. Memoized by DBOS."""
    kwargs = {
        "model": MODEL,
        "input": input_items,
        "tools": [{"type": "web_search"}, SAVE_LEAD_TOOL],
    }
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id
    response = openai_client.responses.create(**kwargs)
    return {
        "id": response.id,
        "output": [item.model_dump() for item in response.output],
    }


@DBOS.workflow()
def research_workflow(topic: str, target_count: int) -> str:
    DBOS.logger.info(f"Researching {topic!r} (target={target_count})")
    input_items: list = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(target_count=target_count),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\n"
                f"Target number of leads: {target_count}\n"
                "Find distinct entries one at a time and save each with save_lead."
            ),
        },
    ]
    previous_response_id: str | None = None
    saved = 0

    for iteration in range(MAX_ITERATIONS):
        response = call_model(input_items, previous_response_id)
        previous_response_id = response["id"]

        tool_outputs: list = []
        for item in response["output"]:
            if item.get("type") == "function_call" and item.get("name") == "save_lead":
                args = json.loads(item["arguments"])
                result = append_lead_to_csv(
                    args["name"], args["description"], args["source_url"]
                )
                saved += 1
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": result,
                    }
                )

        if saved >= target_count or not tool_outputs:
            DBOS.logger.info(
                f"Finished after {iteration + 1} iterations, saved {saved} leads"
            )
            return f"Saved {saved} leads to {CSV_PATH}"

        # Only the tool outputs are sent on subsequent turns; `previous_response_id`
        # threads the full conversation history server-side.
        input_items = tool_outputs

    return f"Reached max iterations; saved {saved} leads to {CSV_PATH}"


class ResearchRequest(BaseModel):
    topic: str
    target_count: int = 10


@app.post("/research")
def start_research(request: ResearchRequest):
    """Kick off a research workflow in the background and return its ID."""
    handle = DBOS.start_workflow(
        research_workflow, request.topic, request.target_count
    )
    return {"workflow_id": handle.workflow_id, "status": "started"}


@app.post("/research/enqueue")
def enqueue_research(request: ResearchRequest):
    """Enqueue a research workflow onto the shared DBOS queue.

    Unlike /research (which starts the workflow immediately), this hands the
    job to `research_queue` — useful when you want managed concurrency, rate
    limiting, or fan-out of many requests without overwhelming your LLM quota.
    """
    handle = research_queue.enqueue(
        research_workflow, request.topic, request.target_count
    )
    return {"workflow_id": handle.workflow_id, "status": "enqueued"}


@app.get("/research/{workflow_id}")
def get_research(workflow_id: str):
    """Fetch the current status (and result, if completed) of a workflow."""
    status = DBOS.retrieve_workflow(workflow_id).get_status()
    return {
        "workflow_id": workflow_id,
        "status": status.status,
        "output": status.output,
    }


@app.get("/research")
def list_research():
    """List recent research workflows."""
    workflows = DBOS.list_workflows(
        name=research_workflow.__qualname__, sort_desc=True, limit=20
    )
    return [
        {"workflow_id": w.workflow_id, "status": w.status, "input": w.input}
        for w in workflows
    ]


if __name__ == "__main__":
    config: DBOSConfig = {
        "name": "workshop",
        "system_database_url": os.environ.get("DBOS_SYSTEM_DATABASE_URL"),
        "conductor_key": os.environ.get("DBOS_CONDUCTOR_KEY"),
    }
    DBOS(config=config)
    DBOS.launch()
    uvicorn.run(app, host="0.0.0.0", port=1234)
