# ContextCore

A personal learning and breakdown tool that takes a financial or tech headline - typed, pasted, or fetched from a URL - and traces its causal chain across **finance**, **macroeconomics**, and **technology**.

**Live app:** https://contextcore.streamlit.app/

---

## What it does

1. Paste a headline, article snippet, or URL
2. The agent classifies the domain (macro / finance / tech / cross-domain) and gathers grounding evidence live from Wikipedia and NewsAPI
3. Two LLM extraction calls turn that evidence into:
   - A **causal chain** — ordered, domain-tagged cause-effect steps with confidence scores
   - A **knowledge graph** — entities and relationships, visualized interactively
4. You can keep asking follow-up questions — the chat reasons over the already-built graph and research rather than re-querying live sources every time

---

<img width="932" height="440" alt="Context core" src="https://github.com/user-attachments/assets/9e15b05e-4e62-411a-a170-b2a0d7d8d314" />





<!-- <img width="1800" height="1800" alt="contextcore_architecture_technical" src="https://github.com/user-attachments/assets/8144f5ed-0dbe-4d96-b9b6-718fa031d67a" /> -->



```

## Setup

```bash
git clone <repo-url>
cd Context_core
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
```

Run locally:
```bash
streamlit run app.py
```

---

## Known limitations

- **Scope is intentionally narrow** — built and tested for finance/tech/macro headlines. Other topics may work but aren't the design target.
- **URL fetching doesn't work on protected sites** — Reuters, WSJ, Bloomberg, and similar publishers block non-browser scraping (401s or dropped connections). Wikipedia and smaller publications generally work fine.
- **Input is capped at 8,000 characters** to keep cost and latency predictable. Cannot process pdf inputs. Long PDFs/documents aren't chunked yet — a full corpus-scale graph (chunk → extract → merge entities) is a known next step, not yet built.
- **The grader occasionally returns inconsistent scores** even on well-grounded answers — an open issue, not fully root-caused. The core causal chain / graph output is reliable; the confidence scoring layer is the least mature part of the pipeline.
- **Follow-up chat sometimes declines to search** for genuinely new information even when instructed to, instead of calling `search_live_news`/`search_wikipedia`

---

## Built with

Python, Streamlit, LlamaIndex, OpenAI API (GPT-4o-mini), Wikipedia API, NewsAPI, streamlit-agraph, Streamlit Community Cloud, GitHub, Novus.ai
