"""
core/extraction.py
-------------------
Two LLM calls that turn raw agent research into structured output:

  1. extract_causal_chain() -- produces a summary + ordered list of
     cause-effect steps (CausalStep shape, matching the original
     Bolt.new prototype's types.ts).

  2. grade_answer() -- the 6-metric LLM-as-judge grader, scoring the
     summary against the actual retrieved chunk text (not just titles --
     this was a real bug we hit and fixed: the grader can't verify claims
     against bare titles, it needs the chunk content).

Kept separate from agent_setup.py because these are pure
input -> LLM call -> structured output functions, no tool-calling
or agent state involved.
"""

import json
from llama_index.core import Settings


CAUSAL_CHAIN_PROMPT_TEMPLATE = """Here is research gathered about a news headline classified as: {domain}

HEADLINE: {headline}

RESEARCH:
{research}

Extract a causal chain of 3-5 steps showing how this headline ripples across
finance, macro, and tech domains. Each step should be a single cause-effect link.

Return ONLY valid JSON in this exact format, no other text:
{{
  "summary": "2-3 sentence plain-language summary of what this means overall",
  "causal_chain": [
    {{
      "domain": "tech" | "finance" | "macro",
      "title": "short step title, 3-6 words",
      "description": "1-2 sentences explaining this step",
      "confidence": 0.0,
      "evidence": "which source/data this step is grounded in"
    }}
  ]
}}

Use only information from the RESEARCH above. Order steps as a logical chain
(first cause, then each downstream effect). confidence should reflect how
directly the RESEARCH supports that specific step (0.0-1.0).
"""


def _strip_json_fences(raw: str) -> str:
    """Remove ```json ... ``` markdown fences if the LLM added them."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].replace("json", "", 1).strip()
    return raw


def extract_causal_chain(headline: str, domain: str, research: str) -> dict:
    """
    Calls the LLM once to convert raw agent research into a structured
    causal chain. Falls back to a flat summary (no chain) if JSON parsing fails,
    so the UI never breaks even on a malformed response.

    Returns: {"summary": str, "causal_chain": list[dict]}
    """
    prompt = CAUSAL_CHAIN_PROMPT_TEMPLATE.format(
        domain=domain, headline=headline, research=research
    )
    response = Settings.llm.complete(prompt)
    raw_json = _strip_json_fences(response.text)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        data = {"summary": research, "causal_chain": []}

    return {
        "summary": data.get("summary", ""),
        "causal_chain": data.get("causal_chain", []),
    }


GRAPH_PROMPT_TEMPLATE = """Here is research gathered about a news headline classified as: {domain}

HEADLINE: {headline}

RESEARCH:
{research}

Extract a knowledge graph showing the entities and relationships in this research.
Identify the key entities (organizations, concepts, indicators, sectors) and how
they connect causally or structurally.

Return ONLY valid JSON in this exact format, no other text:
{{
  "nodes": [
    {{
      "id": "short_lowercase_id",
      "label": "Display Name",
      "domain": "tech" | "finance" | "macro",
      "weight": 1
    }}
  ],
  "edges": [
    {{
      "source": "source_node_id",
      "target": "target_node_id",
      "label": "short relationship verb, e.g. 'raises', 'impacts', 'drives'"
    }}
  ]
}}

Rules:
- 5-9 nodes total. weight is 1-3 (3 = most central to the story, 1 = peripheral).
- Every edge's source/target MUST match a node id exactly.
- Use only information from the RESEARCH above. Don't invent entities not mentioned.
- ids should be short, lowercase, no spaces (use underscores).
"""


def extract_graph(headline: str, domain: str, research: str) -> dict:
    """
    Calls the LLM once to extract a knowledge graph (nodes + edges) from the
    same research used for the causal chain. This is a SEPARATE call from
    extract_causal_chain() -- different output shape, different purpose
    (visual network vs ordered narrative steps).

    Returns: {"nodes": list[dict], "edges": list[dict]}
    Falls back to empty graph if parsing fails, so the UI never breaks.
    """
    prompt = GRAPH_PROMPT_TEMPLATE.format(
        domain=domain, headline=headline, research=research
    )
    response = Settings.llm.complete(prompt)
    raw_json = _strip_json_fences(response.text)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        data = {"nodes": [], "edges": []}

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    # Defensive: drop edges that reference a node id that doesn't exist
    node_ids = {n.get("id") for n in nodes}
    edges = [e for e in edges if e.get("source") in node_ids and e.get("target") in node_ids]

    return {"nodes": nodes, "edges": edges}

GRADER_PROMPT_TEMPLATE = """You are a strict evaluation agent. Your job is to grade an AI-generated answer
against the retrieved source chunks it was based on. You must ONLY use the provided chunks as ground truth --
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

CHAT_FOLLOWUP_PROMPT_TEMPLATE = """You are answering a follow-up question about a news headline you already analyzed.

ORIGINAL HEADLINE: {headline}

RESEARCH (from your earlier analysis):
{research}

KNOWLEDGE GRAPH (entities and relationships from your analysis):
Nodes: {nodes}
Edges: {edges}

CONVERSATION SO FAR:
{history}

FOLLOW-UP QUESTION: {question}

Answer the follow-up using ONLY the research and graph above — do not introduce
new facts not present there. If the graph or research doesn't contain enough
information to answer, say so honestly rather than guessing. Keep the answer
conversational and concise (2-4 sentences).
"""


async def answer_followup_async(headline: str, research: str, graph_data: dict, history: list, question: str) -> str:
    """
    Answers a follow-up chat question using the lightweight chat FunctionAgent
    (see core/agent_setup.py: setup_chat_agent). The agent PREFERS answering
    from the research/graph context already provided, but can call
    search_wikipedia or search_live_news if the question needs information
    genuinely not covered there -- scoped to finance/tech/macro topics.
    """
    from core.agent_setup import setup_chat_agent

    nodes_str = ", ".join([n.get("label", n.get("id", "")) for n in graph_data.get("nodes", [])])
    edges_str = "; ".join([
        f"{e['source']} -{e.get('label','')}-> {e['target']}"
        for e in graph_data.get("edges", [])
    ])
    history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history]) or "(no prior messages)"

    prompt = CHAT_FOLLOWUP_PROMPT_TEMPLATE.format(
        headline=headline,
        research=research,
        nodes=nodes_str,
        edges=edges_str,
        history=history_str,
        question=question,
    )

    chat_agent = setup_chat_agent()
    response = await chat_agent.run(prompt)
    return str(response).strip()

def grade_answer(query: str, domain: str, chunks: list, answer: str) -> dict:
    """
    Scores `answer` against `chunks` on 6 metrics using an LLM-as-judge prompt.
    `chunks` must include actual text content (not just titles) -- chunks
    with only a title give the grader nothing to verify claims against,
    which silently produces all-zero scores (a bug we hit and fixed).
    """
    chunks_text = "\n\n".join([
        f"[Chunk {i + 1} - Source: {c['source']}, Date: {c.get('published_at', 'unknown')}]\n"
        f"Title: {c.get('title', '')}\n"
        f"Content: {c.get('text', '')}"
        for i, c in enumerate(chunks)
    ]) or "No chunks retrieved."

    prompt = GRADER_PROMPT_TEMPLATE.format(
        query=query, domain=domain, chunks=chunks_text, answer=answer
    )
    response = Settings.llm.complete(prompt)
    raw = _strip_json_fences(response.text)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "parse failed", "raw": raw}


def overall_confidence(grading: dict) -> float:
    """Average of the 6 grader metrics, or 0.0 if grading failed."""
    if "error" in grading:
        return 0.0
    keys = [
        "faithfulness", "context_relevance", "causal_validity",
        "cross_domain_coverage", "completeness", "source_recency",
    ]
    return sum(grading.get(k, 0) for k in keys) / len(keys)