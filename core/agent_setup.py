"""
core/agent_setup.py
--------------------
Builds the ContextCore research agent: live tools (Wikipedia, NewsAPI,
domain classifier) wrapped into a LlamaIndex FunctionAgent.

Why FunctionAgent (not ReActAgent):
  ReActAgent relies on text-based "Thought:/Action:" parsing, which proved
  unreliable in testing (documented upstream issue: gpt-4o-mini sometimes
  skips tool calls entirely). FunctionAgent uses native OpenAI function-calling
  (structured tool_calls field), which was reliable across all tests.

Why live tools (no pre-built index):
  Earlier versions used a pre-indexed ChromaDB of ~120 docs (FRED + Wikipedia +
  News snapshot). This had a hard scope limitation -- any headline outside the
  indexed topics returned irrelevant chunks. Switching to live tool calls per
  query removes this limitation entirely.
"""

import os
import re
import requests
import streamlit as st
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI


@st.cache_resource
def get_run_logs() -> dict:
    """
    Thread-safe-ish singleton log of what the agent did during the current run.
    Cached so the SAME object is referenced by both the cached tool closures
    (set up once) and the UI code (runs every Streamlit rerun).
    """
    return {"chunks": [], "calls": [], "domain": "general"}


def _classify_domain_logic(headline: str):
    """Pure domain-classification logic, no side effects -- testable in isolation."""
    h = headline.lower()

    def matches(words, text):
        return any(re.search(r'\b' + w + r'\b', text) for w in words)

    domains = []
    if matches(["fed", "rate", "inflation", "gdp", "unemployment", "recession", "treasury", "monetary"], h):
        domains.append("macro")
    if matches(["stock", "market", "equity", "bond", "bank", "earnings", "valuation", "nasdaq", "s&p"], h):
        domains.append("finance")
    if matches(["chip", "semiconductor", "ai", "tech", "export", "nvidia", "apple", "microsoft", "startup"], h):
        domains.append("tech")
    if not domains:
        domains.append("general")
    if len(domains) > 1:
        domains.append("cross-domain")
    return domains


@st.cache_resource
def setup_agent() -> FunctionAgent:
    """
    Builds and returns the cached FunctionAgent with 3 tools:
      - classify_domain   (local logic, no API call)
      - search_wikipedia   (live Wikipedia API)
      - search_live_news   (live NewsAPI)
    """
    run_logs = get_run_logs()

    def search_wikipedia(query: str) -> str:
        """Fetch background/conceptual context from Wikipedia for entities,
        institutions, or concepts mentioned in the headline (e.g. 'Federal Reserve',
        'Reserve Bank of India', 'yield curve', 'semiconductor industry').
        Use this to explain what something IS, not for current events."""
        run_logs["calls"].append(f"search_wikipedia({query!r})")

        import wikipediaapi
        wiki = wikipediaapi.Wikipedia(user_agent="ContextCore/1.0", language="en")

        page = wiki.page(query)
        if not page.exists():
            return f"No Wikipedia page found for '{query}'."

        text = page.summary[:1000]

        run_logs["chunks"].append({
            "source": "Wikipedia",
            "title": page.title,
            "url": page.fullurl,
            "published_at": "",
            "text": text,
        })

        return f"Wikipedia: {page.title}\n\n{text}"

    def search_live_news(query: str) -> str:
        """Search for recent/breaking news articles."""
        run_logs["calls"].append(f"search_live_news({query!r})")
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": 3,
                "apiKey": os.getenv("NEWS_API_KEY"),
            },
            timeout=10,
        )
        articles = response.json().get("articles", [])
        if not articles:
            return "No recent news found."

        results = []
        for a in articles:
            run_logs["chunks"].append({
                "source": "NewsAPI",
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", "")[:10],
                "text": a.get("description", ""),
            })
            results.append(
                f"[NEWS] {a['title']} ({a.get('publishedAt', '')[:10]}): {a.get('description', '')}"
            )
        return "\n\n".join(results)

    def classify_domain(headline: str) -> str:
        """Classify headline into macro/finance/tech/cross-domain. Call this FIRST."""
        domains = _classify_domain_logic(headline)
        result = ", ".join(domains)
        run_logs["calls"].append(f"classify_domain -> {result}")
        run_logs["domain"] = result
        return f"Domain classification: {result}"

    wikipedia_tool = FunctionTool.from_defaults(
        fn=search_wikipedia,
        name="search_wikipedia",
        description=(
            "Find background/conceptual context for entities, institutions, or concepts "
            "in the headline (e.g. 'Federal Reserve', 'Reserve Bank of India', 'yield curve'). "
            "Use this to explain what something IS. NOT for current events or recent news."
        ),
    )
    news_tool = FunctionTool.from_defaults(
        fn=search_live_news,
        name="search_live_news",
        description="Find recent/breaking news. NOT for definitions or historical data.",
    )
    domain_tool = FunctionTool.from_defaults(
        fn=classify_domain,
        name="classify_domain",
        description="Classify a headline as macro/finance/tech/cross-domain. Call this FIRST.",
    )

    return FunctionAgent(
        tools=[domain_tool, wikipedia_tool, news_tool],
        llm=OpenAI(model="gpt-4o-mini", temperature=0),
        verbose=True,
        max_iterations=10,
        early_stopping_method="generate",
        system_prompt=(
            "You are a financial research assistant. For each query:\n"
            "1. Call classify_domain ONCE.\n"
            "2. Call search_wikipedia at most TWICE with different queries.\n"
            "3. Call search_live_news at most ONCE.\n"
            "4. Synthesise a final answer immediately after -- do NOT loop back to call tools again.\n"
            "Never call the same tool with the same query twice."
        ),
    )