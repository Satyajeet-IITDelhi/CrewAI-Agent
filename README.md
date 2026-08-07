# CrewAI + Firecrawl Demos (powered by Claude Opus 4.8)

Three runnable Python scripts that progress from a single agent to a full
agentic workflow. Every agent reasons with **Claude Opus 4.8**
(`anthropic/claude-opus-4-8`) and uses **Firecrawl** as its web tool.

| Script | Pattern | What it does |
|---|---|---|
| `agent.py` | **Single AI agent** | One agent scrapes a web page and summarizes it. |
| `02_multi_agent.py` | **Multi-agent crew** | A Researcher + a Writer collaborate in sequence. |
| `03_agentic_workflow.py` | **Agentic workflow (Flow)** | A multi-step Flow: scrape → analyze → recommend. |

## 1. Setup

```powershell
# (recommended) create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m uvicorn main:app --reload

# install dependencies
pip install -r requirements.txt
```

## 2. API keys

Keys live in a git-ignored `.env` file (loaded automatically by `config.py`).
You need **two** keys — they are different services:

| Key | Used for |
|---|---|
| `FIRECRAWL_API_KEY` | scraping/searching the web |
| `ANTHROPIC_API_KEY` | Claude (the agents' reasoning) |

The `.env` file already has your Firecrawl key. **Open `.env` and replace
`sk-ant-REPLACE_ME` with your real Anthropic key** (get one at
https://console.anthropic.com/). The scripts raise a clear error until you do.

> Starting fresh elsewhere? Copy `.env.example` to `.env` and fill in both keys.
> `.env` and `instructions.txt` are listed in `.gitignore`, so your keys won't be
> committed. You can still override either key with a shell environment variable.

## 3. Run

```powershell
python agent.py
python 02_multi_agent.py
python 03_agentic_workflow.py
```

## 4. Web app (FastAPI)

`main.py` wraps the single-agent demo in an API with a browser UI, ready to
deploy to an Azure Web App via `.github/workflows/azure-webapp.yml`.

```powershell
uvicorn main:app --reload   # then open http://127.0.0.1:8000
```

| Route | Purpose |
|---|---|
| `GET /` | The UI — paste a URL, get the report |
| `GET /health` | Liveness probe for Azure |
| `POST /summarize` | `{"url": "..."}` → `{"url": ..., "summary": ...}` |

## Notes

- **Model choice:** edit `CLAUDE_MODEL` in `config.py` to switch models —
  `anthropic/claude-sonnet-4-6` is cheaper/faster, `anthropic/claude-haiku-4-5`
  is best for simple, high-volume tasks.
- **`temperature` is intentionally not set.** It was removed on Opus 4.8 and the
  API rejects it; CrewAI omits the parameter when it's unset.
- **Security:** keys live in `.env`, which is git-ignored (along with
  `instructions.txt`), so they won't be committed. Still rotate the Firecrawl key
  if it has ever been shared in plaintext (e.g. it was pasted into a chat).
- If `crewai_tools` complains about the Firecrawl tool signature, upgrade:
  `pip install -U crewai-tools firecrawl-py`.
