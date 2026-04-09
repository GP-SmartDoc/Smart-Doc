from langchain.messages import SystemMessage, HumanMessage
from src.config.model import visualization_model

DESCRIPTION_SYSTEM_PROMPT = """
You convert retrieved document content into a STRUCTURED description
for a Mermaid diagram.

RULES:
- Only use information from context
- Extract steps, entities, relations
- Keep concise and structured
- Do not generate diagram code
- Only generate description

Example:

User request:
"flowchart for login process"

Context:
user enters email and password
system validates credentials
if valid -> dashboard
else -> error message

Output:
Start
User enters email and password
System validates credentials
If valid go to dashboard
Else show error
"""

def description_agent(state:dict):

    context = "\n".join(state.get("retrieved_chunks", []))

    prompt = f"""
User request:
{state.get("user_request")}

Context:
{context}
"""

    result = visualization_model.invoke([
        SystemMessage(content=DESCRIPTION_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    return {
        "llm_calls":1,
        "description": result.content
    }