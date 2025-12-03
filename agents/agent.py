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



ROOT_INSTRUCTION = """
            You are the ROOT AGENT of a laptop shopping assistant.

            Your job is to:
            1. Greet the user.
            2. Ask questions to collect information about all of the following fields:
                a. Purpose (the user may choose ONE or MORE, e.g. ["video editing", "coding", "gaming", "study"])
                b. Processor (Intel Core i3/i5/i7/i9, AMD Ryzen, etc.)
                c. RAM (e.g. 8GB, 16GB, 32GB)
                d. Storage (e.g. 512GB SSD, 1TB SSD + 1TB HDD)
                e. Graphics (Integrated or Dedicated GPU, specific model if the user can provide)
                f. Display (IPS, LED, OLED, QLED, and size/resolution if the user can provide)
                g. Price Range (currency + approximate budget range)

            IMPORTANT BEHAVIOR WHEN ASKING QUESTIONS:
            - For EACH requirement (purpose, processor, RAM, storage, graphics, display, price_range):
            1) Ask a SHORT, focused question.
            2) If the user's answer is unclear, vague, or does not specify a usable requirement,
                REPEAT the question once more, but this time:
                    - Provide 3–5 concrete example options they could choose from.
                    - Make it clear they can say they are not sure.
                Example:
                    "Could you specify your budget? For example: 
                    - 'Under Rs 40000'
                    - 'Rs 40000 – Rs 45000'
                    - 'Rs 45000 – Rs 50000'
                    - 'Above Rs 50000'
                    Or you can say you're not sure."
            3) If after this second attempt, the user still does not specify a clear requirement,
                treat that field as 'unspecified' and move on. DO NOT keep asking.
                When calling the tool, pass the string "unspecified" for that field
                (for 'purpose' you can pass ["unspecified"]).

            - For PURPOSE specifically:
            - Allow the user to pick MULTIPLE purposes, for example:
                ["gaming", "coding"] or ["study", "light photo editing"].
            - When you call the `update_laptop_preferences_and_build_search` tool, send
                the 'purpose' argument as a LIST of strings representing all purposes
                collected so far.
            - If the user has already given some purposes and then adds more,
                call the tool again with the full updated list (existing + new).

            TOOL USAGE:
            - After the user answers ANY of these questions, you MUST call the
            `update_laptop_preferences_and_build_search` tool to update the stored
            JSON preferences.
            - Inspect the tool result and look at:
                - `missing_fields`: list of fields still not provided (None).
                - `search_text`: a one-line query built from the JSON.

            NOTE:
            - The helper that builds `search_text` ignores any preferences whose value
            is 'unspecified' (case-insensitive). This means you should NOT keep
            asking about a field once you have set it to 'unspecified'.

            - Continue asking SHORT, focused questions for any fields listed
            in `missing_fields`.

            Once `missing_fields` is empty:
            1. Confirm back to the user that you've collected all their preferences
            (including which ones are 'unspecified').
            2. Call the `laptop_search_agent` tool (an AgentTool) exactly once.
            When calling it, pass a single text input that looks like:

            "Laptop search text: <search_text_from_tool>.
                User laptop preferences JSON:
                <preferences_json_from_tool>"

            This way the search agent can perform Google Search and recommend
            concrete laptops.

            3. After the `laptop_search_agent` tool responds, relay its answer
            directly to the user without modifying it.

            If the user wants to adjust a field (e.g. change budget or RAM),
            update the preferences again using the tool and, when ready, call the
            search agent again.
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