**Problem Statement** Purchasing a laptop can be challenging for non-technical users who might find it difficult to compare a dense mix of specifications (e.g., CPU, RAM, GPU, Storage, Display, Price) across hundreds of variants. The project aims to assist such users by capturing their requirements through natural conversation and provide real-time laptop recommendations that aligns with their purpose and budget.

This is an interesting project as it showcases how a multi-agent LLM system can solve real-world decision making. From gaining user preferences through conversation, to delivering recommendations with google search grounding, this application can be easily scaled to solve similar or more complex problems across product/service categories.

 **Why are agents the right solution to this problem?** 
- Sequential LLM Agents helps to perform multi-turn user interaction, allowing follow-up question to be asked the user and obtain a clear set of requirements.
- Multiple agents are easily to design and coordinate making the code modular.
- Each agent focuses on a specific responsibility, thus making it easier to debug and increases reusability.
- Easier to integrate external tools (like Google Search) and maintain in-memory state across interactions.
 
**Project Architecture**
![File Structure](images/MultiAgentLaptopRecommendationSystem.png)
![Workflow Diagram](images/project_workflow.png)

**Major Components Explained**
🟦  **Root Agent**

The main conversational controller.

**Responsibilities:**
✔️ Greets the user
✔️ Asks short, focused questions
✔️ Collects 7 laptop requirements: Purpose (supports multiple), Processor, RAM, Storage, Graphics, Display, Price range
✔️ Ensures best-practice questioning pattern: First question →If unclear → Ask again with examples →Still unclear → Mark as "unspecified"

✔️ Updates session preferences through:
*update_laptop_preferences_and_search_text()*


✔️ After all fields are assigned:
Calls laptop_search_agent with the search text generated from the JSON script containing user's laptop preferences

🟩 **update_laptop_preferences_and_search_text()**

A **FunctionTool** used by the root agent.

**Responsibilities:**
Stores the user’s requirements in memory
Converts undefined fields into "unspecified"
Generates the search text used for Google Search

**Returns:**
preferences_json
search_text
missing_fields

🟦 **Laptop Search Agent**

The main recommendation engine.

Responsibilities:
✔️ Interprets the preferences
✔️ Performs Google web search grounding (via **laptop_web_search_agent**)
✔️ Generates 3–5 real laptop recommendations
✔️ Lists real specs, links, prices, advantages
✔️ Ends with: *“Would you like me to compare two of these, or modify specs/price?”*

**Internal Tools Used:**
*laptop_search_tool:*	Provides grounded web data via google_search
*comparison_tool:*	Compares 2+ laptop models
*modify_laptop_requirements:*	Updates preferences and regenerates recommendations

🟧**Comparison Agent** 
A helper agent to compare between two recommended and summarize differences based on CPU, RAM, GPU, Storage, Display, Price, Pros/cons and helps user with a final recommendation between them.

*The root agent never interacts with this agent directly.*

🟨**modify_laptop_requirements()**

Function to help the user update their preferences (e.g.*“Increase the budget to Rs 50000.”*,
*“Modify RAM to 32GB.”*) and regenerates recommendations based on that. It accepts the previous *preferences_json* and updates only the requested fields and rebuilds the search text to trigger a fresh set of recommendations.