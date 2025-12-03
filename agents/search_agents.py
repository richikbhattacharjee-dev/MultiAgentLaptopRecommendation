# search_agents.py

import json
from typing import Optional, Dict, Any, List, Union

from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool

from .comparison_agent import comparison_tool  # NEW

MODEL_ID = "gemini-2.5-flash"

LaptopPreferenceValue = Union[str, List[str], None]


# -------------------------------------------------------------------
# Helpers for building search text from a preference JSON
# (stateless version, used by modify_laptop_requirements)
# -------------------------------------------------------------------

def _is_unspecified(value: Any) -> bool:
    """
    Helper to detect 'unspecified' values (case-insensitive),
    whether they are stored as a string or inside a list.
    """
    if value is None:
        return False

    if isinstance(value, str):
        return value.strip().lower() == "unspecified"

    if isinstance(value, list):
        return (
            len(value) > 0
            and all(
                isinstance(v, str) and v.strip().lower() == "unspecified"
                for v in value
            )
        )

    return False


def _build_search_text_from_preferences(
    prefs: Dict[str, LaptopPreferenceValue]
) -> str:
    """
    Turn the preference JSON into a text query suitable for web search.

    Any preference whose value is 'unspecified' (case-insensitive) is
    ignored when constructing the search text.
    """
    parts: List[str] = []

    # Purpose: can be a single purpose or multiple purposes
    purpose_value = prefs.get("purpose")
    if purpose_value and not _is_unspecified(purpose_value):
        if isinstance(purpose_value, list):
            for p in purpose_value:
                if isinstance(p, str) and p.strip() and p.strip().lower() != "unspecified":
                    parts.append(f"{p.strip()} laptop")
        elif isinstance(purpose_value, str):
            if purpose_value.strip().lower() != "unspecified":
                parts.append(f"{purpose_value.strip()} laptop")

    processor_value = prefs.get("processor")
    if isinstance(processor_value, str) and processor_value.strip() and not _is_unspecified(processor_value):
        parts.append(f"with {processor_value.strip()} processor")

    ram_value = prefs.get("ram")
    if isinstance(ram_value, str) and ram_value.strip() and not _is_unspecified(ram_value):
        parts.append(f"{ram_value.strip()} RAM")

    storage_value = prefs.get("storage")
    if isinstance(storage_value, str) and storage_value.strip() and not _is_unspecified(storage_value):
        parts.append(f"{storage_value.strip()} storage")

    graphics_value = prefs.get("graphics")
    if isinstance(graphics_value, str) and graphics_value.strip() and not _is_unspecified(graphics_value):
        parts.append(f"{graphics_value.strip()} graphics")

    display_value = prefs.get("display")
    if isinstance(display_value, str) and display_value.strip() and not _is_unspecified(display_value):
        parts.append(f"{display_value.strip()} display")

    price_range_value = prefs.get("price_range")
    if isinstance(price_range_value, str) and price_range_value.strip() and not _is_unspecified(price_range_value):
        parts.append(f"within budget {price_range_value.strip()}")

    if not parts:
        return "laptop recommendations"

    return " ".join(parts) + " best laptop recommendations"


# -------------------------------------------------------------------
# NEW: FunctionTool-style modifier for laptop requirements
# -------------------------------------------------------------------

def modify_laptop_requirements(
    preferences_json: str,
    purpose: Optional[List[str]] = None,
    processor: Optional[str] = None,
    ram: Optional[str] = None,
    storage: Optional[str] = None,
    graphics: Optional[str] = None,
    display: Optional[str] = None,
    price_range: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Modify existing laptop requirements.

    This function is designed to be used as an ADK FunctionTool by the
    laptop_search_agent when the user asks to change specs or price range.

    Parameters:
        - preferences_json: JSON string in the same format produced by
          update_laptop_preferences_and_search_text() in agent.py.
        - purpose: OPTIONAL new list of purposes (replace the old one).
        - processor, ram, storage, graphics, display, price_range:
          OPTIONAL new values for each field. Any argument that is not None
          will overwrite the corresponding field in the parsed preferences.

    Behavior:
        - Parses preferences_json into a dict (or creates a default structure
          if parsing fails).
        - Applies any non-None updates.
        - Rebuilds a new search_text based on the updated preferences.
        - Returns:
            {
              "preferences_json": "<updated pretty JSON>",
              "search_text": "<updated one-line search query>"
            }

    NOTE:
        - This function is STATELESS: it does not touch any in-memory global
          preferences. It only operates on the JSON passed in.
    """
    try:
        prefs = json.loads(preferences_json) if preferences_json else {}
    except (json.JSONDecodeError, TypeError):
        prefs = {}

    # Ensure all expected keys exist
    prefs.setdefault("purpose", None)
    prefs.setdefault("processor", None)
    prefs.setdefault("ram", None)
    prefs.setdefault("storage", None)
    prefs.setdefault("graphics", None)
    prefs.setdefault("display", None)
    prefs.setdefault("price_range", None)

    # Apply updates
    if purpose is not None:
        prefs["purpose"] = purpose
    if processor is not None:
        prefs["processor"] = processor
    if ram is not None:
        prefs["ram"] = ram
    if storage is not None:
        prefs["storage"] = storage
    if graphics is not None:
        prefs["graphics"] = graphics
    if display is not None:
        prefs["display"] = display
    if price_range is not None:
        prefs["price_range"] = price_range

    updated_preferences_json = json.dumps(prefs, indent=2)
    search_text = _build_search_text_from_preferences(prefs)

    return {
        "preferences_json": updated_preferences_json,
        "search_text": search_text,
    }


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
#
# Now ALSO capable of:
#   - Comparing two recommended laptops (via comparison_tool)
#   - Modifying requirements (via modify_laptop_requirements) and
#     re-running search with updated preferences.
# -------------------------------------------------------------------
laptop_search_agent = LlmAgent(
    name="laptop_search_agent",
    model=MODEL_ID,
    description=(
        "A specialist laptop recommendation agent that uses a dedicated "
        "web-search grounding agent to find current laptop options and "
        "then returns clear recommendations to the user. It can also "
        "compare two recommended laptops and update requirements to "
        "regenerate recommendations."
    ),
    instruction="""
You are a laptop recommendation specialist.

You will receive a single text input from another agent. That text summarizes the
user's laptop requirements (purpose, CPU, RAM, storage, GPU, display, budget).
It will look roughly like:

  "Laptop search text: <search_text>.
   User laptop preferences JSON:
   <preferences_json>"

You have access to:
- A SEARCH GROUNDING TOOL (laptop_search_tool) which uses `google_search`
  to fetch up-to-date laptop/product information.
- A COMPARISON TOOL (comparison_tool) that compares two or more concrete
  laptop models.
- A REQUIREMENT MODIFICATION TOOL (modify_laptop_requirements) that
  updates the user's preferences JSON and produces a new search_text.

Your main tasks:

1. FIRST RESPONSE (after being invoked by the root agent)
   1.1. Read and understand the laptop requirements from the input.
   1.2. Use the search grounding capability to obtain relevant, up-to-date
        laptop options (specific models with specs and prices).
   1.3. Based on the grounded information, produce a clear, user-friendly
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
        - Explain in 1–2 short sentences why each model fits the requirements.
   1.4. At the END of this message, explicitly ask:

        "Would you like me to:
         (a) compare any two of these laptops side by side, or
         (b) change the specs or your price range and see updated options?"

2. FOLLOW-UP INTERACTIONS

   a) If the user asks for a COMPARISON between two recommended laptops:
      - Identify which two models they care about (by name or index).
      - Call the comparison_tool with a short prompt that:
          * Clearly lists the two (or more) models to compare.
          * Optionally describes the user's priorities (e.g. gaming, battery).
      - Use the tool's response as the core of your reply.
      - Present the comparison in a user-friendly way (e.g. bullet points or
        a mini table) and, if appropriate, recommend one.

   b) If the user asks to MODIFY their requirements (spec changes or budget):
      - Extract the last known preferences_json from the conversation
        (the one originally provided in your input).
      - Call modify_laptop_requirements with:
          * preferences_json
          * ONLY the fields the user wants to change (leave others as None).
      - Take the returned `preferences_json` and `search_text` and use the
        search grounding tool again to get a fresh set of recommendations.
      - Present 3–5 updated recommendations using the same format as in the
        first response.
      - Again, end by offering:
        "Do you want a comparison between any two of these, or further changes?"

CRITICAL BEHAVIOR:
- You are talking to the END USER (through the root agent).
- ALWAYS generate a helpful, concrete final answer in each message.
- You MAY ask brief, focused follow-up questions ONLY to:
    * Clarify which two laptops to compare, OR
    * Clarify which requirement(s) to modify (e.g. "Only the budget, or RAM too?").
- Avoid re-collecting all requirements from scratch; only change what the
  user asks to change.
- DO NOT expose internal tool calls or the existence of helper agents.
- Your output should be directly displayable to the user as-is.
""",
    tools=[
        laptop_search_tool,       # web-search grounding via google_search
        comparison_tool,          # NEW: comparison helper agent
        modify_laptop_requirements  # NEW: FunctionTool-style modifier
    ],
)
