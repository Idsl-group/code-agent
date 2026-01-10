from typing import Literal
from langgraph.graph import StateGraph, START, END

from schemas import AgentState
from nodes import set_api_key
from nodes import ToolCallingAgent, ReflectionAgent

# --- Graph Construction ---

def build_graph(api_key):
    set_api_key(api_key)
    tool_calling_agent = ToolCallingAgent(api_key)
    reflection_agent = ReflectionAgent(api_key)
    
    workflow = StateGraph(AgentState) # State object will follow the AgentState schema.

    # Add the nodes
    workflow.add_node("tool_calling_node", tool_calling_agent.tool_calling_node)
    workflow.add_node("tool_node", tool_calling_agent.tool_node)
    workflow.add_node("reflection_node", reflection_agent.reflect)
    workflow.add_node("user_input_node", reflection_agent.user_input_node)
    # Set the entry point
    workflow.add_edge(START, "tool_calling_node")
    workflow.add_edge("tool_node", "reflection_node")
    workflow.add_edge("user_input_node", "tool_calling_node")
    # Add the conditional edges
    workflow.add_conditional_edges(
        "tool_calling_node",
        tool_calling_agent.should_continue,
        {
            "tool_node": "tool_node",
            "reflect": "reflection_node",
            "__end__": END,
        }
    )
    workflow.add_conditional_edges(
        "reflection_node",
        reflection_agent.should_continue,
        {
            "tool_calling_node": "tool_calling_node", 
            "user_input_node": "user_input_node",
             "__end__": END, 
        },
    )
    
    coding_agent_graph = workflow.compile()
    print("\n Graph compiled successfully!")
    print(" Reflection loop enabled: reflection_node can route back to tool_calling_node")
    print("="*60 + "\n")
    return coding_agent_graph