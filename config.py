"""
Shared configuration for all three CrewAI demos.

- Keys are loaded from a local .env file (see .env.example). The Firecrawl key
  falls back to instructions.txt if not found in the environment.
- The LLM that powers every agent is pluggable. Two providers are wired up:
    * "groq"      -> Llama 3.3 70B via Groq (fast, free tier)  [DEFAULT]
    * "anthropic" -> Claude Opus 4.8 (most capable, paid)
  Pick one with LLM_PROVIDER in .env. Firecrawl scrapes the web; the LLM does the
  reasoning — they are two different services with two different keys.
"""

import os
import sys

from crewai import LLM
from dotenv import load_dotenv

# Load keys from the local .env file into the environment (no-op if .env is missing).
load_dotenv()

# --- Windows console: print CrewAI's emoji log lines as UTF-8 instead of cp1252.
# Without this, the event-bus log handlers raise "'charmap' codec can't encode
# character '\U0001f680'" on every step (cosmetic, but noisy). Safe no-op on POSIX.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

# --- Groq compatibility shim for CrewAI's prompt-cache markers.
# Newer CrewAI stamps a `cache_breakpoint` flag onto system/user messages for
# Anthropic-style prompt caching. Native providers strip it before the API call,
# but the LiteLLM fallback path (the only route for Groq, which has no native
# provider) forwards it verbatim — and Groq rejects the unknown property with a
# 400. We make the marker a no-op so the key is never added. This loses prompt
# caching (which Groq doesn't support anyway) but lets Groq work end to end.
try:
    import crewai.llms.cache as _crewai_cache

    _crewai_cache.mark_cache_breakpoint = lambda message: message  # type: ignore[assignment]
except Exception:  # pragma: no cover - defensive; never block startup on the shim
    pass

# Which LLM provider to use: "groq" (default) or "anthropic".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Model strings carry a provider prefix because CrewAI routes through LiteLLM.
GROQ_MODEL = "groq/openai/gpt-oss-120b"  # swap for e.g. groq/llama-3.1-8b-instant
CLAUDE_MODEL = "anthropic/claude-opus-4-8"   # or anthropic/claude-sonnet-4-6


def load_firecrawl_key() -> str:
    """Return the Firecrawl key from the env var, or fall back to instructions.txt."""
    key = os.getenv("FIRECRAWL_API_KEY")
    if not key:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "instructions.txt"), encoding="utf-8") as f:
            # instructions.txt may hold several lines; take the first fc-... token.
            key = next(
                (line.strip() for line in f if line.strip().startswith("fc-")), ""
            )
        os.environ["FIRECRAWL_API_KEY"] = key
    return key


def get_llm() -> LLM:
    """Build the LLM that every agent will use for reasoning, per LLM_PROVIDER."""
    if LLM_PROVIDER == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        _require_key(
            api_key,
            name="ANTHROPIC_API_KEY",
            placeholder="sk-ant-REPLACE_ME",
            console="https://console.anthropic.com/",
        )
        # NOTE: don't pass `temperature` for Opus 4.8 — it was removed on this model
        # and the API returns a 400 if sent. CrewAI omits it when unset.
        return LLM(model=CLAUDE_MODEL, api_key=api_key)

    if LLM_PROVIDER == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        _require_key(
            api_key,
            name="GROQ_API_KEY",
            placeholder="gsk_REPLACE_ME",
            console="https://console.groq.com/keys",
        )
        return LLM(model=GROQ_MODEL, api_key=api_key)

    raise RuntimeError(
        f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. Set it to 'groq' or 'anthropic' in .env."
    )


def _require_key(value: str | None, *, name: str, placeholder: str, console: str) -> None:
    if not value:
        raise RuntimeError(
            f"{name} is not set.\n"
            f"Add it to your .env file (see .env.example) or set it in the shell.\n"
            f"Get a key from {console}\n"
            "(Firecrawl scrapes the web; the LLM does the thinking — you need both keys.)"
        )
    if value.strip() in ("", placeholder):
        raise RuntimeError(
            f"{name} is still the placeholder value. Edit .env and put your real key "
            f"in place of '{placeholder}'."
        )


# Groq's free tier allows 8,000 tokens per minute on gpt-oss-120b. A full page can be
# 15K+ tokens, so cap the scraped text. Scraped markdown is link-heavy (~2-2.5 chars per
# token), so 6,000 chars ≈ 2.5-3K tokens, leaving room for the agent's prompt, its
# reasoning history and the answer. 12,000 still overflowed on Azure (10.9K tokens).
MAX_SCRAPE_CHARS = 6_000


def make_scrape_tool():
    """Firecrawl scrape tool that truncates long pages to fit the LLM's token budget."""
    from crewai_tools import FirecrawlScrapeWebsiteTool

    class TrimmedFirecrawlScrapeTool(FirecrawlScrapeWebsiteTool):
        def _run(self, url: str):
            result = super()._run(url)
            text = getattr(result, "markdown", None) or str(result)
            if len(text) > MAX_SCRAPE_CHARS:
                text = text[:MAX_SCRAPE_CHARS] + "\n\n[... page truncated ...]"
            return text

    return TrimmedFirecrawlScrapeTool()
