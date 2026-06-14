"""
Minimal test comparing ReActAgent vs FunctionAgent tool-calling behaviour.
Run with: python test_react.py
Requires OPENAI_API_KEY in environment.
"""
from dotenv import load_dotenv
load_dotenv()
import asyncio
import os

from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

# ---------------------------------------------------------------------------
# One simple tool — answer is ONLY knowable via this function
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Return the current weather for a given city. This is the ONLY source of weather data."""
    data = {
        "zanzibar": "Sunny, 31°C, humidity 78%",
        "nuuk": "Blizzard, -18°C, wind 60 km/h",
    }
    return data.get(city.lower(), f"No weather data available for {city!r}.")

weather_tool = FunctionTool.from_defaults(fn=get_weather)

QUERY = "What is the weather like in Nuuk right now? Use the tool."
LLM = OpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# 1. Legacy ReActAgent (llama_index.core.agent)
# ---------------------------------------------------------------------------
def test_react_legacy():
    print("\n" + "="*60)
    print("TEST 1 — Legacy ReActAgent (llama_index.core.agent)")
    print("="*60)
    from llama_index.core.agent import ReActAgent
    agent = ReActAgent.from_tools([weather_tool], llm=LLM, verbose=True, max_iterations=5)
    response = agent.chat(QUERY)
    print("\n>>> FINAL RESPONSE:", response)


# ---------------------------------------------------------------------------
# 2. Workflow-based ReActAgent (llama_index.core.agent.workflow)
# ---------------------------------------------------------------------------
async def test_react_workflow():
    print("\n" + "="*60)
    print("TEST 2 — Workflow ReActAgent (llama_index.core.agent.workflow)")
    print("="*60)
    from llama_index.core.agent.workflow import ReActAgent as WFReActAgent
    agent = WFReActAgent(tools=[weather_tool], llm=LLM, verbose=True)
    response = await agent.run(QUERY)
    print("\n>>> FINAL RESPONSE:", response)


# ---------------------------------------------------------------------------
# 3. FunctionAgent (llama_index.core.agent.workflow)
# ---------------------------------------------------------------------------
async def test_function_agent():
    print("\n" + "="*60)
    print("TEST 3 — FunctionAgent (llama_index.core.agent.workflow)")
    print("="*60)
    from llama_index.core.agent.workflow import FunctionAgent
    agent = FunctionAgent(tools=[weather_tool], llm=LLM, verbose=True)
    response = await agent.run(QUERY)
    print("\n>>> FINAL RESPONSE:", response)


# ---------------------------------------------------------------------------
# 4. FunctionAgent with system_prompt nudge (belt-and-suspenders)
# ---------------------------------------------------------------------------
async def test_function_agent_forced():
    print("\n" + "="*60)
    print("TEST 4 — FunctionAgent + explicit system prompt")
    print("="*60)
    from llama_index.core.agent.workflow import FunctionAgent
    agent = FunctionAgent(
        tools=[weather_tool],
        llm=LLM,
        verbose=True,
        system_prompt=(
            "You MUST call the available tools to answer questions. "
            "Never answer from your own knowledge."
        ),
    )
    response = await agent.run(QUERY)
    print("\n>>> FINAL RESPONSE:", response)


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY first.")

    # Legacy sync agent
    try:
        test_react_legacy()
    except Exception as e:
        print(f"[ERROR] Legacy ReActAgent: {e}")

    # Async agents
    async def run_all():
        for fn in (test_react_workflow, test_function_agent, test_function_agent_forced):
            try:
                await fn()
            except Exception as e:
                print(f"[ERROR] {fn.__name__}: {e}")

    asyncio.run(run_all())
