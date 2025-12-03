
from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool

MODEL_ID = "gemini-2.5-flash"


# -------------------------------------------------------------------
# Helper agent: only responsible for calling google_search to fetch
# laptop/product information from the web.
# -------------------------------------------------------------------
laptop_web_search_agent = LlmAgent(
    name="laptop_web_search_agent",
    model=MODEL_ID,
    description=(
        "Helper agent that uses the google_search tool to fetch grounded, "
        "up-to-date information about laptop products."
    ),
    tools=[google_search],
    instruction="""
You are a helper agent that is used ONLY for web search grounding.

You are NOT talking directly to the end user. You are invoked by another
agent that needs grounded, up-to-date information on laptops and related
products.

You will receive a short text input that contains a laptop-related search
query or a description of the user's needs (e.g. 'best 16GB RAM gaming
laptops under $1200').

Your job:

1. Interpret that input as a search query.
2. Call the `google_search` tool AT LEAST ONCE with a suitable query that
   will return product pages, comparison articles, or reviews for specific
   laptop models.
3. Focus on getting results that mention concrete laptops (brand + model),
   their specifications, and typical prices.

Your RESPONSE should be a concise textual summary of the most relevant
search results, focusing on:

- Laptop model names (brand + model)
- Key specs (CPU, RAM, storage, GPU, display)
- Price indications
- Any particularly useful product / review URLs

IMPORTANT:
- Do NOT address the end user in a conversational way. Your output is
  ONLY used as grounding information by another agent.
- Do NOT ask questions or request clarification.
- ALWAYS call `google_search` at least once before responding.
""",
)

laptop_search_tool = AgentTool(agent=laptop_web_search_agent)


# -------------------------------------------------------------------
# Main laptop search / recommendation agent.
# Uses laptop_web_search_agent as its search_grounding_tool so that
# it can rely on grounded web results but still return a single,
# final answer directly to the user.
# -------------------------------------------------------------------
laptop_search_agent = LlmAgent(
    name="laptop_search_agent",
    model=MODEL_ID,
    description=(
        "A specialist laptop recommendation agent that uses a dedicated "
        "web-search grounding agent to find current laptop options and "
        "then returns clear recommendations to the user."
    ),
    instruction="""
You are a laptop recommendation specialist.

You will receive a single text input from another agent. That text summarizes the
user's laptop requirements (purpose, CPU, RAM, storage, GPU, display, budget).
It will look roughly like:

  "Laptop search text: <search_text>.
   User laptop preferences JSON:
   <preferences_json>"

You have access to a SEARCH GROUNDING TOOL, which is itself another agent that
uses the `google_search` tool to fetch up-to-date laptop/product information
from the web. Use this grounding capability to base your recommendations on
current real-world products.

Your job:

1. Read and understand the laptop requirements from the input.
2. Use the search grounding capability to obtain relevant, up-to-date laptop
   options (specific models with specs and prices).
3. Based on the grounded information, produce a clear, user-friendly FINAL
   answer that:
   - Lists 3–5 recommended laptop models (brand + model name).
   - For each, briefly mention:
       * Processor
       * RAM
       * Storage
       * Graphics capability
       * Display type/size (if available)
       * Typical price range (approximate)
       * Optionally one useful product or review link
   - Explain in 1–2 short sentences why each model fits the given requirements.

4. Close by asking the user if they want to refine the recommendations
   (e.g. cheaper, lighter, stronger GPU, better battery, etc.).

CRITICAL BEHAVIOR:
- You are talking to the END USER (through the root agent).
- ALWAYS generate a helpful, concrete final answer in a SINGLE message.
- DO NOT ask the user for more details; treat the provided preferences as
  sufficient to recommend.
- DO NOT expose internal tool calls or the existence of the helper agent.
- Your output should be directly displayable to the user as-is.
""",
    # No direct tools here; all web search is done via the grounding agent:
    tools=[laptop_search_tool],
)
