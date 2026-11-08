"""
Shared configuration for all three CrewAI demos.

- Keys are loaded from a local .env file (see .env.example). The Firecrawl key
  falls back to instructions.txt if not found in the environment.
- The LLM that powers every agent is pluggable. Three providers are wired up:
    * "groq"      -> gpt-oss-120b via Groq (free tier: 8K tokens/min)  [DEFAULT]
    * "azure"     -> Azure OpenAI deployment (gpt-4.1-mini), pay-per-use
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

# Which LLM provider to use: "groq" (default), "azure" or "anthropic".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Model strings carry a provider prefix so CrewAI can route them.
GROQ_MODEL = "groq/openai/gpt-oss-120b"  # swap for e.g. groq/llama-3.1-8b-instant
CLAUDE_MODEL = "anthropic/claude-opus-4-8"   # or anthropic/claude-sonnet-4-6
# Azure OpenAI: the part after "azure/" is the *deployment* name in your resource.
# Project-specific AZURE_LLM_* names on purpose: the generic AZURE_OPENAI_* variables
# are often already set machine-wide for other projects, and .env never overrides
# existing environment variables.
AZURE_DEPLOYMENT = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4.1-mini")
AZURE_API_VERSION = os.getenv("AZURE_LLM_API_VERSION", "2024-10-21")


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

    if LLM_PROVIDER == "azure":
        api_key = os.getenv("AZURE_LLM_API_KEY")
        _require_key(
            api_key,
            name="AZURE_LLM_API_KEY",
            placeholder="your-azure-openai-key",
            console="Azure portal > your Foundry/OpenAI resource > Keys and Endpoint",
        )
        endpoint = os.getenv("AZURE_LLM_ENDPOINT")
        _require_key(
            endpoint,
            name="AZURE_LLM_ENDPOINT",
            placeholder="https://your-resource.openai.azure.com/",
            console="Azure portal > your Foundry/OpenAI resource > Keys and Endpoint",
        )
        # Native Azure provider (needs crewai[azure-ai-inference]); it takes
        # `endpoint`, not `base_url`.
        return LLM(
            model=f"azure/{AZURE_DEPLOYMENT}",
            api_key=api_key,
            endpoint=endpoint,
            api_version=AZURE_API_VERSION,
        )

    raise RuntimeError(
        f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. "
        "Set it to 'groq', 'azure' or 'anthropic' in .env."
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


# Cap the scraped text so it fits the LLM's token budget.
# - Groq free tier: 8,000 tokens/min on gpt-oss-120b. Scraped markdown is link-heavy
#   (~2-2.5 chars per token), so 6,000 chars ≈ 2.5-3K tokens, leaving room for the
#   prompt, reasoning history and answer (12,000 overflowed at 10.9K tokens).
# - Azure OpenAI / Anthropic: far larger limits, so keep much more of the page.
MAX_SCRAPE_CHARS = 6_000 if LLM_PROVIDER == "groq" else 20_000


def make_scrape_tool():
    """Firecrawl scrape tool that truncates long pages to fit the LLM's token budget."""
    from crewai_tools import FirecrawlScrapeWebsiteTool

    class TrimmedFirecrawlScrapeTool(FirecrawlScrapeWebsiteTool):
        def _run(self, url: str):
            result = super()._run(url)
            text = getattr(result, "markdown", None) or str(result)
            if len(text) > MAX_SCRAPE_CHARS:
                # Wording matters: a bare "truncated" note made the agent re-scrape the
                # same URL looking for the rest, pushing runs past the MCP gateway's
                # ~50 s timeout.
                text = text[:MAX_SCRAPE_CHARS] + (
                    "\n\n[End of scraped content. This is all the content available "
                    "for this page; do not scrape it again. Write the summary now.]"
                )
            return text

    return TrimmedFirecrawlScrapeTool()
