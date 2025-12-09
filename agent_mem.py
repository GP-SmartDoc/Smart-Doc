# agent2.py
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

# import memory system (correct version lives here)
from Memory_Manager import MemoryManager, collection, embedding_model

# initialize memory manager
memory_manager = MemoryManager(collection, embedding_model)

# Initialize LLM
model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key="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"
)

class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    memory_context: str

def process(state: AgentState) -> AgentState:
    user_msg = state["messages"][-1].content

    # query memory
    results = memory_manager.query_memory(user_msg)
    retrieved_docs = results.get("documents", [[]])[0]
    memory_text = "\n".join(retrieved_docs) if retrieved_docs else "No memory."

    system_context = (
        "You have access to long-term memory.\n"
        f"Relevant memory:\n{memory_text}\n"
    )

    msgs = [{"role": "system", "content": system_context}, *state["messages"]]

    response = model.invoke(msgs)
    ai_msg = AIMessage(content=response.content)

    # SAVE memory
    memory_manager.add_memory(user_msg)

    return {
        "messages": state["messages"] + [ai_msg],
        "memory_context": memory_text
    }

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

def run_agent(user_text: str):
    state = {"messages": [HumanMessage(content=user_text)], "memory_context": ""}
    result = agent.invoke(state)
    return result["messages"][-1].content

def main():
    print("Memory Agent is running...")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Agent: Goodbye!")
            break

        response = run_agent(user_input)
        print("Agent:", response)


if __name__ == "__main__":
    main()