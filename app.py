import nest_asyncio
nest_asyncio.apply()
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="ContextCore", layout="wide")


st.title("📰 ContextCore")
st.caption("Paste a financial/tech headline — get the causal chain across finance, macro, and tech")

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
[data-testid="stAppViewContainer"] > .main {
    order: 1;
}
section[data-testid="stSidebar"] {
    order: 2;
}
[data-testid="stAppViewContainer"] {
    display: flex;
    flex-direction: row;
}
</style>
""", unsafe_allow_html=True)

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import chromadb

@st.cache_resource
def load_index():
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    
    chroma_client     = chromadb.PersistentClient(path="./utils/chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("contextcore")
    vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context   = StorageContext.from_defaults(vector_store=vector_store)
    
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

index = load_index()
# st.success("Index loaded — ready for queries")

import requests
import re
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

# Global log of chunks retrieved during the current run
retrieved_chunks_log = []
tool_calls_log = []
domain_log = []

@st.cache_resource
def setup_agent():
    retriever = index.as_retriever(similarity_top_k=3)

    def search_knowledge_base(query: str) -> str:
        """Search for macro indicators, background concepts, and indexed news."""
        tool_calls_log.append(f"Searched knowledge base: '{query}'")
        nodes = retriever.retrieve(query)
        results = []
        for n in nodes:
            meta = n.metadata
            retrieved_chunks_log.append({
                "source": meta.get("source", "unknown"),
                "title": meta.get("title", meta.get("series", meta.get("series_id",""))),
                "url": meta.get("url", ""),
                "published_at": meta.get("published_at", ""),
            })
            results.append(f"[{meta.get('source','unknown')}]: {n.text[:300]}")
        return "\n\n".join(results)

    def search_live_news(query: str) -> str:
        """Search for recent/breaking news articles."""
        tool_calls_log.append(f"Searched live news: '{query}'")
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": query, "language": "en", "sortBy": "relevancy", "pageSize": 3, "apiKey": os.getenv("NEWS_API_KEY")}
        )
        articles = response.json().get("articles", [])
        if not articles:
            return "No recent news found."
        results = []
        for a in articles:
            retrieved_chunks_log.append({
                "source": "NewsAPI",
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", "")[:10],
            })
            results.append(f"[NEWS] {a['title']} ({a.get('publishedAt','')[:10]}): {a.get('description','')}")
        return "\n\n".join(results)

    def classify_domain(headline: str) -> str:
        """Classify headline into macro/finance/tech/cross-domain."""
        h = headline.lower()
        def matches(words, text):
            return any(re.search(r'\b' + w + r'\b', text) for w in words)
        domains = []
        if matches(["fed","rate","inflation","gdp","unemployment","recession","treasury","monetary"], h):
            domains.append("macro")
        if matches(["stock","market","equity","bond","bank","earnings","valuation","nasdaq","s&p"], h):
            domains.append("finance")
        if matches(["chip","semiconductor","ai","tech","export","nvidia","apple","microsoft","startup"], h):
            domains.append("tech")
        if not domains:
            domains.append("general")
        if len(domains) > 1:
            domains.append("cross-domain")

        tool_calls_log.append(f"Classified domain: {', '.join(domains)}")
        domain_log.append(", ".join(domains))
        return f"Domain classification: {', '.join(domains)}"

    knowledge_tool = FunctionTool.from_defaults(fn=search_knowledge_base, name="search_knowledge_base",
        description="Find economic indicators (rates, CPI, GDP, unemployment), background concepts, and indexed news. NOT for live breaking news.")
    news_tool = FunctionTool.from_defaults(fn=search_live_news, name="search_live_news",
        description="Find recent/breaking news. NOT for definitions or historical data.")
    domain_tool = FunctionTool.from_defaults(fn=classify_domain, name="classify_domain",
        description="Classify a headline as macro/finance/tech/cross-domain. Call this FIRST.")

#     SYSTEM_PROMPT = """You are a financial news analyst with access to tools for retrieving 
# real economic data and news. You MUST use your tools before answering — never answer from 
# memory alone.

# For every query:
# 1. ALWAYS call classify_domain first.
# 2. ALWAYS call search_knowledge_base to get macro/background data.
# 3. ALWAYS call search_live_news to get recent articles.

# After using all three tools, provide your final response starting with the word "Answer:" 
# followed by exactly these three section headers:

# Answer:
# CAUSAL_CHAIN:
# [2-4 sentences explaining the cause-and-effect chain across finance/macro/tech domains, based on retrieved data]

# BACKGROUND:
# [2-3 sentences of context the reader needs to understand this event, based on retrieved data]

# DRY_FACTS:
# [2-3 sentences stating only the verified facts/numbers from retrieved data, no interpretation]
# """

    return ReActAgent(
        tools=[domain_tool, knowledge_tool, news_tool],
        llm=Settings.llm,
        verbose=True,
        max_iterations=10,
        # system_prompt=SYSTEM_PROMPT,
    )

agent = setup_agent()
# st.success("Agent ready")

import json

# Track which tools were called, for the sidebar activity log
tool_calls_log = []

GRADER_PROMPT = """You are a strict evaluation agent. Your job is to grade an AI-generated answer 
against the retrieved source chunks it was based on. You must ONLY use the provided chunks as ground truth — 
do NOT use your own knowledge to judge factual correctness.

ORIGINAL QUERY:
{query}

DOMAIN CLASSIFICATION:
{domain}

RETRIEVED CHUNKS (the only allowed sources of truth):
{chunks}

AGENT'S ANSWER:
{answer}

Score the answer on these 6 metrics, each from 0.0 to 1.0:

1. faithfulness: Are claims in the answer supported by the retrieved chunks?
2. context_relevance: Are the retrieved chunks actually relevant to the query?
3. causal_validity: Does the cause-effect reasoning follow logically?
4. cross_domain_coverage: Does the answer address all domains in the classification?
5. completeness: Does the answer address all parts of the query?
6. source_recency: Are "what happened" claims based on recent sources vs only old background?

Return ONLY valid JSON, no other text:
{{
  "faithfulness": 0.0,
  "context_relevance": 0.0,
  "causal_validity": 0.0,
  "cross_domain_coverage": 0.0,
  "completeness": 0.0,
  "source_recency": 0.0,
  "flagged_claims": [],
  "notes": ""
}}
"""

def grade_answer(query, domain, chunks, answer):
    chunks_text = "\n\n".join([
        f"[Chunk {i+1} - Source: {c['source']}, Date: {c.get('published_at','unknown')}]\n{c.get('title','')}"
        for i, c in enumerate(chunks)
    ]) or "No chunks retrieved."

    prompt = GRADER_PROMPT.format(query=query, domain=domain, chunks=chunks_text, answer=answer)
    response = Settings.llm.complete(prompt)
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].replace("json", "", 1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "parse failed", "raw": raw}


def parse_sections(text):
    sections = {"CAUSAL_CHAIN": "", "BACKGROUND": "", "DRY_FACTS": ""}
    pattern = r"(CAUSAL_CHAIN|BACKGROUND|DRY_FACTS):\s*(.*?)(?=(CAUSAL_CHAIN|BACKGROUND|DRY_FACTS):|$)"
    for match in re.finditer(pattern, text, re.DOTALL):
        sections[match.group(1)] = match.group(2).strip()
    return sections


def bar_color(score):
    if score >= 0.8:
        return "var(--color-text-success)" if False else "#1D9E75"
    elif score >= 0.5:
        return "#EF9F27"
    else:
        return "#E24B4A"


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.divider()
headline = st.text_area("Paste a headline or short article snippet", height=100,
    placeholder="e.g. Fed raises interest rates by 0.25% amid persistent inflation")

if st.button("Analyze", type="primary"):
    if not headline.strip():
        st.warning("Please enter a headline first.")
    else:
        retrieved_chunks_log.clear()
        tool_calls_log.clear()

        with st.spinner("Researching..."):
            import asyncio
            async def run_agent(query):
                full_query = f"{query}\n\nWhat does this mean? Explain the causal chain across finance, macro, and tech domains."
                return await agent.run(full_query)

            response = asyncio.run(run_agent(headline))
            raw_answer = str(response)

            FORMAT_PROMPT = f"""Here is an analysis of a news headline:

{raw_answer}

Reformat this into EXACTLY these three sections, using only information from the analysis above 
(do not add new facts):

CAUSAL_CHAIN:
[2-4 sentences on the cause-and-effect chain]

BACKGROUND:
[2-3 sentences of context]

DRY_FACTS:
[2-3 sentences of verified facts/numbers only, no interpretation]
"""
            format_response = Settings.llm.complete(FORMAT_PROMPT)
            answer_text = format_response.text
            sections = parse_sections(answer_text)

        domain_str = domain_log[-1] if domain_log else "general"
        grading = grade_answer(headline, domain_str, retrieved_chunks_log, answer_text)

        main_col = st.container()

        with main_col:
            st.subheader("What this means")
            st.write(sections["CAUSAL_CHAIN"] or "_No causal chain returned._")

            with st.expander("Background"):
                st.write(sections["BACKGROUND"] or "_No background returned._")

            with st.expander("Dry facts and sources", expanded=False):
                st.write(sections["DRY_FACTS"] or "_No dry facts returned._")
                st.markdown("**Sources:**")
                seen = set()
                for c in retrieved_chunks_log:
                    key = c.get("url") or c.get("title")
                    if key in seen or not c.get("title"):
                        continue
                    seen.add(key)
                    if c.get("url"):
                        st.markdown(f"- [{c['source']}] [{c['title']}]({c['url']})")
                    else:
                        st.markdown(f"- [{c['source']}] {c['title']}")

        with st.sidebar:
            st.markdown("**Agent activity**")
            for t in tool_calls_log:
                st.caption(t)
            st.caption(f"Retrieved {len(retrieved_chunks_log)} chunks")
            st.caption(f"Tool calls: {len(tool_calls_log)}")
            for t in tool_calls_log:
                st.caption(f"  - {t}")

            st.markdown("**Confidence**")
            if "error" not in grading:
                for metric in ["faithfulness","context_relevance","causal_validity","cross_domain_coverage","completeness","source_recency"]:
                    score = grading.get(metric, 0)
                    st.caption(f"{metric.replace('_',' ').title()}: {score}")
                    st.progress(score)
            else:
                st.caption("Grading failed")

        with st.expander("Raw agent output (debug)"):
            st.code(answer_text)