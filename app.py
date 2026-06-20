"""
app.py
------
ContextCore -- Streamlit UI.

All agent/extraction/grading logic lives in core/ -- this file only
handles layout, user input, and rendering.

Rendering reads from st.session_state.analysis (set once per "Analyze" click)
so that chat follow-ups -- which trigger their own Streamlit reruns -- don't
lose the displayed analysis.

Layout: input pane (1/3 width) on the left, results pane (2/3 width,
Chat/Graph/Causal Chain tabs) on the right.
"""

import nest_asyncio
try:
    nest_asyncio.apply()
except ValueError:
    # nest_asyncio can't patch uvloop (used by Streamlit Cloud's server).
    # Safe to skip -- uvloop handles nested loops differently and this
    # patch isn't needed in that environment.
    pass

import asyncio
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components

load_dotenv()

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from core.agent_setup import setup_agent, get_run_logs
from core.extraction import (
    extract_causal_chain,
    extract_graph,
    grade_answer,
    overall_confidence,
    answer_followup_async,
)
from core.utils import bar_color
from core.document import extract_text_from_url

from streamlit_agraph import agraph, Node, Edge, Config

# -------------------------------------------------------------
# Page config + styling
# -------------------------------------------------------------

st.set_page_config(page_title="ContextCore", layout="wide")

# -------------------------------------------------------------
# Pendo SDK
# -------------------------------------------------------------

components.html("""
<script>
(function(apiKey){
    var parent = window.parent;
    if (parent._pendoSnippetLoaded) return;
    parent._pendoSnippetLoaded = true;
    (function(p,e,n,d,o){var v,w,x,y,z;o=p[d]=p[d]||{};o._q=o._q||[];
    v=['initialize','identify','updateOptions','pageLoad','track','trackAgent'];for(w=0,x=v.length;w<x;++w)(function(m){
    o[m]=o[m]||function(){o._q[m===v[0]?'unshift':'push']([m].concat([].slice.call(arguments,0)));};})(v[w]);
    y=e.createElement(n);y.async=!0;y.src='https://cdn.pendo.io/agent/static/'+apiKey+'/pendo.js';
    z=e.getElementsByTagName(n)[0];z.parentNode.insertBefore(y,z);})(parent,parent.document,'script','pendo');

    parent.pendo.initialize({
        visitor: { id: '' }
    });
})('37371830-fa96-40d1-afbc-07c6facf51e3');
</script>
""", height=0)

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
}
[data-testid="stAppViewContainer"] > .main {
    order: 1;
}
section[data-testid="stSidebar"] {
    order: 2;
}
[data-testid="stTextArea"] textarea {
    padding: 16px;
}
[data-testid="stAppViewContainer"] {
    display: flex;
    flex-direction: row;
}
[data-testid="stTextAreaInstructions"] {
    display: none;
}
[data-testid="stTabs"] {
    max-height: 600px;
    overflow-y: auto;
}
[data-testid="stRadio"] {
    margin-bottom: -10px;
}
[data-testid="stHorizontalBlock"] {
    gap: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("ContextCore")
# st.caption("Paste a financial/tech headline — get the causal chain across finance, macro, and tech")

DOMAIN_COLORS = {"tech": "#5B8DEF", "finance": "#2DD4A8", "macro": "#F0A857", "general": "#888888"}
MAX_INPUT_CHARS = 8000

# -------------------------------------------------------------
# Model + agent setup (cached)
# -------------------------------------------------------------

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)

agent = setup_agent()
run_logs = get_run_logs()

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# st.divider()

# -------------------------------------------------------------
# Layout: 1/3 input pane, 2/3 results pane
# -------------------------------------------------------------

input_col, results_col = st.columns([1, 2])

# ---------------------------------------------------------
# Input pane
# ---------------------------------------------------------

with input_col:
    input_mode = st.radio("Input", ["Type/paste text", "Paste a URL"], horizontal=True)

    if input_mode == "Type/paste text":
        headline = st.text_area(
            "Paste a headline or short article snippet",
            height=120,
            max_chars=MAX_INPUT_CHARS,
            placeholder="e.g. Fed raises interest rates by 0.25% amid persistent inflation",
        )
        st.caption(f"{len(headline)} / {MAX_INPUT_CHARS} characters")
    else:
        url_input = st.text_input("Paste an article URL", placeholder="https://...")
        headline = ""
        if url_input:
            try:
                result = extract_text_from_url(url_input)
                headline = result["text"]
                if result["truncated"]:
                    st.info(f"Article was {result['original_length']} characters — using first {MAX_INPUT_CHARS}.")
                else:
                    st.success(f"Extracted {len(headline)} characters from the article.")
                with st.expander("Preview extracted text"):
                    st.write(headline)
            except Exception as e:
                st.error(f"Could not fetch article: {e}")

    if st.button("Analyze", type="primary"):
        if not headline.strip():
            st.warning("Please enter a headline first.")
        else:
            run_logs["chunks"].clear()
            run_logs["calls"].clear()
            run_logs["domain"] = "general"

            with st.spinner("Researching..."):
                async def run_agent(query: str):
                    return await agent.run(query)

                agent_output = asyncio.run(run_agent(headline))
                raw_answer = str(agent_output)

                domain_str = run_logs["domain"]
                extracted = extract_causal_chain(headline, domain_str, raw_answer)
                summary = extracted["summary"]
                causal_chain = extracted["causal_chain"]

                graph_data = extract_graph(headline, domain_str, raw_answer)

            chunks = run_logs["chunks"]
            grading = grade_answer(headline, domain_str, chunks, summary)

            # Save everything needed for rendering + follow-up chat into session state
            st.session_state.analysis = {
                "headline": headline,
                "summary": summary,
                "causal_chain": causal_chain,
                "graph_data": graph_data,
                "raw_answer": raw_answer,
                "chunks": chunks,
                "grading": grading,
                "tool_calls": list(run_logs["calls"]),
            }
            st.session_state.chat_history = []  # reset chat on new headline

# ---------------------------------------------------------
# Results pane (reads from session_state, survives chat reruns)
# ---------------------------------------------------------

with results_col:
    if st.session_state.analysis:
        a = st.session_state.analysis

        overall_conf = overall_confidence(a["grading"])
        conf_color = bar_color(overall_conf)

        # unique_sources = {c["source"]: c for c in a["chunks"]}.values()
        # source_chips_html = "".join([
        #     f'<span style="font-size:11px;padding:4px 10px;border-radius:10px;'
        #     f'background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);'
        #     f'color:#ccc">{c["source"]}</span>'
        #     for c in unique_sources
        # ])

        # st.markdown(f"""
        # <div style="display:flex;align-items:center;gap:16px;background:rgba(255,255,255,0.03);
        #             border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:14px 18px;margin-bottom:16px">
        #     <div style="flex-shrink:0">
        #         <div style="font-size:24px;font-weight:700;color:{conf_color}">{overall_conf:.0%}</div>
        #         <div style="font-size:11px;color:#888;margin-top:2px">confidence</div>
        #     </div>
        #     <div style="flex:1">
        #         <div style="background:#2a2a2a;border-radius:4px;height:6px;overflow:hidden">
        #             <div style="background:{conf_color};width:{overall_conf*100}%;height:6px;border-radius:4px"></div>
        #         </div>
        #         <div style="font-size:11px;color:#888;margin-top:4px">Based on {len(a["chunks"])} retrieved sources</div>
        #     </div>
        #     <div style="display:flex;flex-wrap:wrap;gap:6px">
        #         {source_chips_html}
        #     </div>
        # </div>
        # """, unsafe_allow_html=True)

        tab_chat, tab_graph, tab_chain = st.tabs(["Chat", "Graph", "Causal Chain"])

        # ── Chat tab ──
        with tab_chat:
            chat_box = st.container(height=340)
            with chat_box:
                with st.chat_message("assistant"):
                    st.write(a["summary"])

                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

            followup = st.chat_input("Ask a follow-up about this headline...")
            if followup:
                st.session_state.chat_history.append({"role": "user", "content": followup})

                async def run_followup():
                    return await answer_followup_async(
                        a["headline"], a["raw_answer"], a["graph_data"],
                        st.session_state.chat_history[:-1], followup,
                    )

                reply = asyncio.run(run_followup())
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

            # with st.expander("Dry facts and sources", expanded=False):
            #     st.markdown("**Sources:**")
            #     seen = set()
            #     for c in a["chunks"]:
            #         key = c.get("url") or c.get("title")
            #         if key in seen or not c.get("title"):
            #             continue
            #         seen.add(key)
            #         if c.get("url"):
            #             st.markdown(f"- [{c['source']}] [{c['title']}]({c['url']})")
            #         else:
            #             st.markdown(f"- [{c['source']}] {c['title']}")

        # ── Graph tab ──
        with tab_graph:
            nodes_raw = a["graph_data"].get("nodes", [])
            edges_raw = a["graph_data"].get("edges", [])

            if nodes_raw:
                max_weight_node = max(nodes_raw, key=lambda n: n.get("weight", 1))
                center_id = max_weight_node.get("id")

                agraph_nodes = [
                    Node(
                        id=n["id"],
                        label=n.get("label", n["id"]),
                        size=12 + (n.get("weight", 2) * 6),
                        color=DOMAIN_COLORS.get(n.get("domain", "general"), "#888888"),
                        font={"size": 13, "color": "#ffffff", "strokeWidth": 0},
                        **({"x": 0, "y": 0} if n["id"] == center_id else {}),
                    )
                    for n in nodes_raw
                ]
                agraph_edges = [
                    Edge(
                        source=e["source"],
                        target=e["target"],
                        label=e.get("label", ""),
                        font={"size": 10, "color": "#aaaaaa", "strokeWidth": 4, "strokeColor": "#0e1117"},
                        color="#555555",
                    )
                    for e in edges_raw
                ]
                config = Config(
                    width="100%",
                    height=350,
                    directed=True,
                    physics={
                        "barnesHut": {
                            "gravitationalConstant": -8000,
                            "centralGravity": 0.5,
                            "springLength": 180,
                            "springConstant": 0.04,
                        },
                        "minVelocity": 0.75,
                    },
                    hierarchical=False,
                    node={"borderWidth": 2, "borderWidthSelected": 4},
                )
                agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)

                st.markdown(
                    f'<div style="display:flex;gap:16px;margin-top:8px;font-size:12px;color:#888">'
                    f'<span><span style="color:{DOMAIN_COLORS["tech"]}">●</span> Tech</span>'
                    f'<span><span style="color:{DOMAIN_COLORS["finance"]}">●</span> Finance</span>'
                    f'<span><span style="color:{DOMAIN_COLORS["macro"]}">●</span> Macro</span>'
                    f'</div>', unsafe_allow_html=True
                )
            else:
                st.write("_No graph data returned._")

        # ── Causal Chain tab ──
        # ── Causal Chain tab ──
        with tab_chain:
            chain_box = st.container(height=380)
            with chain_box:
                if a["causal_chain"]:
                    for i, step in enumerate(a["causal_chain"], 1):
                        domain_label = step.get("domain", "general").title()
                        conf = step.get("confidence", 0)
                        color = bar_color(conf)
                        st.markdown(
                            f"""<div style="border-left:3px solid {color};padding:8px 12px;margin-bottom:8px">
                            <strong>{i}. {step.get('title', '')}</strong>
                            <span style="float:right;color:{color};font-size:12px">{domain_label} - {conf:.0%}</span><br>
                            <span style="font-size:14px">{step.get('description', '')}</span><br>
                            <span style="font-size:11px;color:gray;font-style:italic">{step.get('evidence', '')}</span>
                            </div>""",
                            unsafe_allow_html=True
                        )
                else:
                    st.write("_No causal chain returned._")
    else:
        tab_chat, tab_graph, tab_chain = st.tabs(["Chat", "Graph", "Causal Chain"])
        with tab_chat:
            st.caption("Your causal-chain summary and follow-up conversation will appear here.")
        with tab_graph:
            st.caption("An interactive knowledge graph of entities and relationships will appear here.")
        with tab_chain:
            st.caption("A step-by-step causal chain, grounded in retrieved sources, will appear here.")

# -------------------------------------------------------------
# Sidebar: technical detail (currently disabled for submission)
# -------------------------------------------------------------

# with st.sidebar:
#     if st.session_state.analysis:
#         a = st.session_state.analysis
#         st.markdown("**Agent activity**")
#         for t in a["tool_calls"]:
#             st.caption(t)
#         st.caption(f"Retrieved {len(a['chunks'])} chunks")
#
#         st.markdown("**Confidence**")
#         if "error" not in a["grading"]:
#             for metric in ["faithfulness", "context_relevance", "causal_validity",
#                             "cross_domain_coverage", "completeness", "source_recency"]:
#                 score = a["grading"].get(metric, 0)
#                 st.caption(f"{metric.replace('_', ' ').title()}: {score}")
#                 color = bar_color(score)
#                 st.markdown(
#                     f"""<div style="background:#2a2a2a;border-radius:4px;height:6px;margin-bottom:8px">
#                     <div style="background:{color};width:{score*100}%;height:6px;border-radius:4px"></div>
#                     </div>""",
#                     unsafe_allow_html=True
#                 )
#         else:
#             st.caption("Grading failed")
#             st.code(a["grading"].get("raw", "no raw output"))
#
#         with st.expander("Raw agent output (debug)"):
#             st.code(a["raw_answer"])