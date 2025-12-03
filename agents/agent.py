import json
from typing import Optional, Dict, Any, List, Union

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool

from .search_agents import laptop_search_agent

MODEL_ID = "gemini-2.5-flash"


# -------------------------------------------------------------------
# Simple in-memory "session" for this process
# (ADK sessions will keep conversation context; this JSON stores
# laptop-specific preferences for the current chat session.)
# -------------------------------------------------------------------

# Allow purpose to be a single string or a list of strings (multiple purposes) 
LaptopPreferenceValue = Union[str, List[str], None]

_laptop_preferences: Dict[str, LaptopPreferenceValue] = {
    "purpose": None,       # can be str, List[str], or None
    "processor": None,
    "ram": None,
    "storage": None,
    "graphics": None,
    "display": None,
    "price_range": None,
}


def _is_unspecified(value: Any) -> bool:
    """
    Helper to detect 'unspecified' values (case-insensitive), whether
    they are stored as a string or inside a list.
    """
    if value is None:
        return False

    if isinstance(value, str):
        return value.strip().lower() == "unspecified"

    if isinstance(value, list):
        # If all entries in the list are unspecified, treat the whole thing
        # as unspecified for search-text purposes.
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
    Internal helper to turn the preference JSON into a text query
    suitable for web search.

    Any preference whose value is 'unspecified' (case-insensitive) is
    ignored when constructing the search text.
    """
    parts: List[str] = []

    # Purpose: can be a single purpose or multiple purposes
    purpose_value = prefs.get("purpose")
    if purpose_value and not _is_unspecified(purpose_value):
        if isinstance(purpose_value, list):
            # e.g. ["gaming", "coding"] -> "gaming laptop", "coding laptop"
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

    # Fallback if nothing is set yet or everything is 'unspecified'
    if not parts:
        return "laptop recommendations"

    # Build a reasonably natural search text
    return " ".join(parts) + " best laptop recommendations"


def update_laptop_preferences_and_search_text(
    purpose: Optional[List[str]] = None,
    processor: Optional[str] = None,
    ram: Optional[str] = None,
    storage: Optional[str] = None,
    graphics: Optional[str] = None,
    display: Optional[str] = None,
    price_range: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stores/updates the user's laptop preferences in a JSON object and
    builds a search text from those preferences.

    This function is designed to be used as an ADK FunctionTool.

    Parameters:
        - purpose: a list of one or more purposes, e.g. ["gaming", "coding"].
                   The agent is expected to send a list when the user has
                   multiple purposes. If the user cannot specify clearly after
                   being prompted twice, the agent should pass ["unspecified"].
        - processor, ram, storage, graphics, display, price_range:
                   Strings representing user preferences. If the user cannot
                   specify clearly after being prompted twice, the agent should
                   pass "unspecified" for that field.

    Any argument that is not None will overwrite the stored value in the
    in-memory preferences for the current agent process.

    While generating the search text, any preference whose value is
    'unspecified' (case-insensitive) will be ignored.

    Returns:
        {
          "preferences_json": "<pretty JSON string of all preferences>",
          "search_text": "<one-line search query>",
          "missing_fields": ["purpose", "processor", ...]  # still None
        }
    """
    # Update in-memory "session" state
    if purpose is not None:
        # Purpose is expected to be a list of strings
        _laptop_preferences["purpose"] = purpose
    if processor is not None:
        _laptop_preferences["processor"] = processor
    if ram is not None:
        _laptop_preferences["ram"] = ram
    if storage is not None:
        _laptop_preferences["storage"] = storage
    if graphics is not None:
        _laptop_preferences["graphics"] = graphics
    if display is not None:
        _laptop_preferences["display"] = display
    if price_range is not None:
        _laptop_preferences["price_range"] = price_range

    preferences_json = json.dumps(_laptop_preferences, indent=2)

    # Build search text using the helper
    search_text = _build_search_text_from_preferences(_laptop_preferences)

    # A field is considered "missing" only if it is truly unset (None),
    # not when it is 'unspecified'. That way the agent stops re-asking it.
    missing_fields = [
        key for key, value in _laptop_preferences.items() if value is None
    ]

    return {
        "preferences_json": preferences_json,
        "search_text": search_text,
        "missing_fields": missing_fields,
    }



ROOT_INSTRUCTION = ROOT_INSTRUCTION = """
You are the ROOT AGENT of a laptop shopping assistant.

Your responsibilities in the conversation:

-----------------------------------------------------------------------
1. WELCOME + REQUIREMENT COLLECTION
-----------------------------------------------------------------------

Your job is to greet the user and then ask short, focused questions
to collect information about ALL of the following laptop requirements:

   a. Purpose (the user may choose ONE or MULTIPLE purposes, e.g.
      ["video editing", "coding", "gaming", "study"])

   b. Processor (Intel Core i3/i5/i7/i9, AMD Ryzen 5/7/9, Apple M-Series)

   c. RAM (e.g. 8GB, 16GB, 32GB)

   d. Storage (e.g. 512GB SSD, 1TB SSD)

   e. Graphics (Integrated or Dedicated GPU, specific model if possible)

   f. Display (IPS, LED, OLED, QLED, size/resolution if possible)

   g. Price Range (currency + approximate budget range)


-----------------------------------------------------------------------
2. QUESTIONING RULES — VERY IMPORTANT
-----------------------------------------------------------------------

For EACH requirement, follow this pattern:

1) FIRST ATTEMPT:
   Ask ONE short, clear question for the field.

2) IF THE USER'S ANSWER IS UNCLEAR:
   Ask ONLY ONE follow-up question for that field.
   This second question MUST include:
      - 3–5 example options they can choose from
      - An explicit “not sure” option

   Example:
     "Could you specify your budget? For example:
      - Under ₹40,000
      - ₹40,000 – ₹50,000
      - ₹50,000 – ₹70,000
      - Above ₹70,000
      Or you can say you’re not sure."

3) IF STILL UNCLEAR AFTER SECOND ATTEMPT:
   Stop asking about that field.
   Treat it as "unspecified".
   Pass "unspecified" when calling the update tool.
   (For purpose pass ["unspecified"]).


-----------------------------------------------------------------------
3. SPECIAL RULES FOR PURPOSE
-----------------------------------------------------------------------

- The user may specify MULTIPLE purposes.
- When calling the tool, always pass purpose as a LIST of strings.
- If the user later adds more purposes, call the tool again with the
  FULL merged list of purposes.


-----------------------------------------------------------------------
4. TOOL USE DURING REQUIREMENT COLLECTION
-----------------------------------------------------------------------

After the user answers ANY requirement:

You MUST call:
    update_laptop_preferences_and_search_text

This tool updates stored JSON preferences and returns:

    - preferences_json
    - search_text
    - missing_fields

A field is “missing” ONLY if it is None,
NOT if it is "unspecified".

Continue asking short questions for any fields listed in missing_fields.


-----------------------------------------------------------------------
5. WHEN ALL FIELDS ARE COMPLETE
-----------------------------------------------------------------------

When missing_fields becomes empty:

1. Confirm to the user that all preferences have been captured
   (including fields marked "unspecified").

2. Then call the laptop_search_agent ONCE with:

   "Laptop search text: <search_text>.
    User laptop preferences JSON:
    <preferences_json>"

3. DO NOT modify or summarize the laptop_search_agent response.
   Relay it directly to the user.


-----------------------------------------------------------------------
6. AFTER RECOMMENDATIONS (HANDLED BY laptop_search_agent)
-----------------------------------------------------------------------

Once called, the laptop_search_agent will:

- Provide 3–5 grounded laptop recommendations, then ask:

  "Would you like me to:
   (a) compare any two of these laptops side by side, or
   (b) change the specs or your price range and see updated options?"

Comparison and requirement modification are handled internally by:
- comparison_tool
- modify_laptop_requirements

You, the root agent, DO NOT perform comparison logic.


-----------------------------------------------------------------------
7. IF THE USER RETURNS TO THE ROOT AGENT LATER
-----------------------------------------------------------------------

If the user returns saying they want to:

- change RAM / processor / GPU / display
- change price range
- add/remove purposes
- modify any requirement

THEN:

1. Call update_laptop_preferences_and_search_text again
   with ONLY the updated fields.

2. Once missing_fields is empty again,
   call laptop_search_agent again with the updated preferences.

Do NOT recollect all fields unless the user explicitly requests a reset.


-----------------------------------------------------------------------
8. ABSOLUTE RULES
-----------------------------------------------------------------------

- ALWAYS ask short, focused questions.
- NEVER ask more than twice for the same field.
- NEVER re-ask a field marked “unspecified”.
- ALWAYS call the update tool after each user answer.
- ALWAYS delegate search to laptop_search_agent when ready.
- NEVER reveal tool usage or internal agents.
- NEVER modify laptop_search_agent output.


-----------------------------------------------------------------------
END OF ROOT INSTRUCTION
-----------------------------------------------------------------------
"""


# Agent to start the conversation with the user.
# Collects 
root_agent = LlmAgent(
    name="root_agent",
    model=MODEL_ID,
    description=(
        "Root laptop assistant that collects user requirements, stores them "
        "in JSON, builds a search text, and delegates to a Google-Search-based "
        "laptop recommendation agent."
    ),
    instruction=ROOT_INSTRUCTION,
    tools=[update_laptop_preferences_and_search_text],
    sub_agents=[laptop_search_agent]
)