"""
FastAPI wrapper around the CrewAI single-agent demo (see agent.py).

Endpoints:
  GET  /           -> the browser UI (static/index.html)
  GET  /health     -> liveness probe for Azure
  POST /summarize  -> {"url": "..."} -> scrape + summarize with the agent

Run locally:  uvicorn main:app --reload
"""

import os
import sys
from functools import lru_cache

# --- sqlite3 shim for Azure App Service. MUST run before importing crewai.
# CrewAI imports chromadb at module load, and chromadb requires sqlite3 >= 3.35.
# The App Service Python 3.12 image ships an older system sqlite3, so the worker
# dies on import with "unsupported version of sqlite3". pysqlite3-binary bundles a
# current SQLite; swapping it into sys.modules makes chromadb pick it up. Wrapped
# in try/except because the wheel is Linux-only -- on Windows/macOS the stdlib
# sqlite3 is already new enough and this is a no-op.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from crewai import Agent, Crew, Process, Task
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import get_llm, load_firecrawl_key

app = FastAPI(title="CrewAI Web Research API", version="1.0.0")

INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")


class SummarizeRequest(BaseModel):
    url: str


@lru_cache(maxsize=1)
def get_crew() -> Crew:
    """Build the crew once, on first request.

    Deliberately lazy: constructing the tool/LLM needs API keys, and doing it at
    import time would crash the worker on startup (and fail Azure's health probe)
    instead of returning a readable error from the endpoint.
    """
    load_firecrawl_key()
    from crewai_tools import FirecrawlScrapeWebsiteTool

    analyst = Agent(
        role="Web Research Analyst",
        goal="Read a given web page and explain what it says in clear, plain language.",
        backstory=(
            "You are a meticulous analyst who scrapes web pages and turns dense "
            "content into tight, accurate summaries that anyone can understand."
        ),
        tools=[FirecrawlScrapeWebsiteTool()],
        llm=get_llm(),
        verbose=True,
    )
    task = Task(
        description=(
            "Scrape the web page at {url} using your Firecrawl tool. "
            "Then write a summary of the page's main points."
        ),
        expected_output=(
            "A short report with:\n"
            "- A one-sentence description of what the page is about\n"
            "- 3 to 5 bullet points covering the key information\n"
            "- Any notable links, products, or calls to action mentioned"
        ),
        agent=analyst,
    )
    return Crew(
        agents=[analyst], tasks=[task], process=Process.sequential, verbose=True
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Sync `def` on purpose: crew.kickoff() blocks, so FastAPI runs it in a
# threadpool rather than stalling the event loop.
@app.post("/summarize")
def summarize(req: SummarizeRequest) -> dict:
    try:
        result = get_crew().kickoff(inputs={"url": req.url})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"url": req.url, "summary": str(result)}
