"""
DEMO 1 — Single AI Agent
========================
The simplest pattern: ONE agent, ONE task, ONE tool.

A "Web Research Analyst" agent is given the Firecrawl scrape tool and asked to
read a web page and produce a clean summary. The agent (powered by Claude Opus
4.8) decides on its own when to call Firecrawl, reads the scraped content, and
writes the answer.

Run:  python agent.py
"""

from crewai import Agent, Crew, Process, Task

from config import get_llm, load_firecrawl_key, make_scrape_tool

# 1. Make sure the Firecrawl key is available, then build the scrape tool
#    (truncates long pages so they fit the LLM's token budget).
load_firecrawl_key()
scrape_tool = make_scrape_tool()

# 2. The reasoning brain for the agent.
llm = get_llm()

# 3. Define the single agent.
research_analyst = Agent(
    role="Web Research Analyst",
    goal="Read a given web page and explain what it says in clear, plain language.",
    backstory=(
        "You are a meticulous analyst who scrapes web pages and turns dense "
        "content into tight, accurate summaries that anyone can understand."
    ),
    tools=[scrape_tool],
    llm=llm,
    verbose=True,
)

# 4. Define the task for that agent. {url} is filled in at kickoff time.
summarize_task = Task(
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
    agent=research_analyst,
)

# 5. Wrap it in a (one-agent) crew and run it.
crew = Crew(
    agents=[research_analyst],
    tasks=[summarize_task],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    result = crew.kickoff(inputs={"url": "https://www.firecrawl.dev"})
    print("\n\n=========== FINAL RESULT ===========\n")
    print(result)
