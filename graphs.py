from typing import Literal
from schemas import AgentState
from nodes import tool_calling_node, tool_node, set_api_key
from langgraph.graph import StateGraph, START, END

# --- Routing Logic ---

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """
    Determines the next step based on the Agent's last message.
    If the Agent made a tool call, we route to 'tools'.
    Otherwise, we finish (END).
    """
    messages = state["messages"]
    print(f"Messages Length: {len(messages)}")
    print([f.type for f in messages])
    last_message = messages[-1]
    
    # If the LLM output contains tool_calls, route to tools
    if last_message.tool_calls:
        return "tools"
    
    # If no tool calls, we are done
    return END

# --- Graph Construction ---

def build_graph(api_key):
    set_api_key(api_key)
    workflow = StateGraph(AgentState)

    # Add the nodes
    workflow.add_node("tool_calling_node", tool_calling_node)
    workflow.add_node("tools", tool_node)
    
    # Set the entry point
    workflow.set_entry_point("tool_calling_node")

    # Add the conditional edges
    workflow.add_conditional_edges(
        "tool_calling_node",
        should_continue,
        # {
        #     "tools": "tools",
        #     END: END
        # }
    )

    # Add the edge from tools back to agent (The Loop)
    # After a tool runs, control goes back to the agent to process the result

    return workflow.compile()