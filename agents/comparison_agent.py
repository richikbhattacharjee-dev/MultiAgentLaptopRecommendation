# comparison_agent.py

from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool

MODEL_ID = "gemini-2.5-flash"


# -------------------------------------------------------------------
# Comparison agent: compares two or more laptop models using web search
# -------------------------------------------------------------------
comparison_agent = LlmAgent(
    name="comparison_agent",
    model=MODEL_ID,
    description=(
        "Helper agent that compares two or more laptop models using the "
        "google_search tool to fetch grounded, up-to-date information."
    ),
    tools=[google_search],
    instruction="""
You are a helper agent that is used to COMPARE laptop models.

You are NOT talking directly to the end user. You are invoked by another
agent that needs a clear comparison between two (or more) specific laptops.

INPUT:
- A short text prompt that:
    * Names at least two concrete laptop models (brand + model), and
    * May mention the user's priorities (e.g. gaming, portability, battery).

YOUR JOB:

1. For EACH laptop model mentioned in the input, call the `google_search`
   tool at least once with a query that will return product pages or reviews
   for that specific model.

2. From the search results, extract and compare:
   - CPU
   - RAM
   - Storage
   - GPU / graphics capability
   - Display (size, resolution, panel type) if available
   - Typical price range
   - Any standout pros/cons (e.g. thermals, build quality, battery life)

3. Produce a concise, structured comparison that the calling agent can pass
   directly to the user. For example:
   - Short introduction sentence.
   - Bullet-point comparison or a clear section per model.
   - A brief "Which one to choose?" recommendation based on the stated
     priorities (e.g. "For heavier gaming, X; for portability and battery, Y.").

IMPORTANT:
- Do NOT ask questions or request clarification.
- Do NOT address the calling agent; write as if the text will be shown
  directly to the end user.
- ALWAYS call `google_search` at least once per laptop model before responding.
""",
)

comparison_tool = AgentTool(agent=comparison_agent)
