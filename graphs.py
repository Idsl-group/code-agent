import os, json
from os.path import isfile, isdir, join
from typing import Literal
from IPython.display import Image

from langgraph.graph import StateGraph, START, END

from schemas import AgentState
from nodes import set_api_key
from nodes import ToolCallingAgent, ReflectionAgent

# --- Graph Construction ---

def get_latest_version(graphs, new_version_flag=False):
    if graphs is None:
        return "{:.2}".format(0.1)
    version_keys = [float(f) for f in graphs.keys()]
    if len(version_keys)>0:
        version_keys = sorted(version_keys)
        if new_version_flag:
            new_version = version_keys[-1]+0.1
            return "{:.2}".format(new_version)
        return "{:.2}".format(version_keys[-1])
    else:
        return "{:.2}".format(0.1)

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
    workflow.add_edge("user_input_node", "reflection_node")
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
    
    print("\nGraph compiled successfully!")
    graph_viz = coding_agent_graph.get_graph()

    # print Mermaid text
    mermaid_xml = graph_viz.draw_mermaid()
    print(mermaid_xml)
    
    new_version = False
    if isfile(os.getenv("GRAPH_PATH")):
        with open(os.getenv("GRAPH_PATH"), "r") as fp:
            graphs = json.load(fp)
        if graphs[get_latest_version(graphs)]==mermaid_xml:
            new_version = False
        else:
            new_version = True
            graphs[get_latest_version(graphs, new_version_flag=True)] = mermaid_xml
            with open(os.getenv("GRAPH_PATH"), "w") as fp:
                json.dump(graphs, fp)
    else:
        new_version = True
        graphs = {get_latest_version(None): mermaid_xml}
        with open(os.getenv("GRAPH_PATH"), "w") as fp:
            json.dump(graphs, fp)

    # this creates graph.png using Mermaid
    if new_version:
        png_bytes = graph_viz.draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_bytes)

    print("\n Graph compiled successfully!")
    print(" Reflection loop enabled: reflection_node can route back to tool_calling_node")
    print("="*60 + "\n")
    return coding_agent_graph