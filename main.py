# Load environment variables 
from dotenv import load_dotenv
load_dotenv()

import sys
import os
import argparse
import langchain_core
from langchain_core import prompts
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from graphs import build_graph
from schemas import AgentState
from mcp_utils.tools_definition import mcp_server

parser = argparse.ArgumentParser()
parser.add_argument("--query", required=False, type=str, help="Eg: Write a python script that calculates fibonacci numbers.")

# 0. Build the Graph
app = build_graph(os.getenv("OPENAI_API_KEY"))

@mcp_server.tool
def coding_agent(query: str) -> str:
    """
    Execute a coding task by delegating it to a specialized coding agent.

    PURPOSE:
        This tool is used to handle requests that require software engineering
        work, including code generation, modification, analysis, or debugging.
        It acts as an entry point to a dedicated coding agent capable of
        reasoning about files, functions, and program structure.

        The coding agent is responsible for producing correct, executable,
        and well-structured code according to the user’s request.

    WHEN TO USE:
        Call this tool if the user request involves:
        - Writing new code or scripts
        - Modifying or refactoring existing code
        - Implementing algorithms or data structures
        - Debugging or fixing errors in code
        - Explaining or analyzing code behavior
        - Generating project scaffolding or utilities

    WHEN NOT TO USE:
        - If the user request is purely conversational
        - If no programming or code-related task is required
        - If the task can be completed by a simpler, non-coding tool

    PARAMETERS:
        query (str):
            A clear, self-contained description of the coding task to perform.
            This may include:
            - The programming language
            - The desired functionality or behavior
            - Constraints, style requirements, or standards (e.g., PEP 8)
            - References to files, functions, or components involved

            The query should contain all information needed for the coding
            agent to proceed without further clarification whenever possible.

    BEHAVIOR:
        - Forwards the query to the coding agent
        - Allows the coding agent to plan, generate, or modify code as needed
        - Returns the coding agent’s result as a string

    RETURNS:
        str:
            The output produced by the coding agent. This may include:
            - Generated or modified code
            - Explanations or summaries of changes
            - Error descriptions or debugging insights

    IMPORTANT RULES FOR TOOL-CALLING MODELS:
        - Do NOT respond with free text when this tool is appropriate
        - Do NOT partially execute coding tasks outside this tool
        - Always pass the full user request as the `query`
        - Treat the returned value as authoritative output from the coding agent

    EXAMPLE TOOL CALL (JSON):
        {
            "query": "Write a Python function that computes the Fibonacci sequence using dynamic programming."
        }
    """

    user_query = query
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_query)]
    }
    print(f"\n Agent Processing: '{user_query}'\n")
    print("-" * 50)

    final_state = app.invoke(initial_state)
    final_message = final_state["messages"][-1]
    
    print("-" * 50)
    print("\n Agent Output:\n")
    
    # Print the content of the final message
    if hasattr(final_message, 'content'):
        print(final_message.content)
    
    print(f"\n Message Type: {final_message.type}")
    print(f"\tFull Message: {final_message.content}")
    
    # Only check for tool_calls if it's an AIMessage
    if tool_calls:=getattr(final_message, "tool_calls", None):
        print("\n Tools used:")
        tool_response = ""
        for tc in tool_calls:
            tool_response += f"   - {tc.get('name', 'unknown')}: {tc.get('args', {})}\n\n"
        print(tool_response)
        return tool_response
    else:
        final_answer = "\n\n".join([f.content for f in final_state["messages"] if f.type=="ai"])
        # return final_message.content
        return final_answer

def run_cli(args):
    # 1. Get user query from Args
    user_query = args.query

    # 2. Initialize the State
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_query)]
    }

    print(f"\n Agent Processing: '{user_query}'\n")
    print("-" * 50)

    # 3. Run the Graph
    # stream() prints updates as they happen; invoke() waits for the end.
    # Using invoke for simpler final output in this script.
    final_state = app.invoke(initial_state)

    # 4. Output the final result
    final_message = final_state["messages"][-1]
    
    print("-" * 50)
    print("\n Agent Output:\n")
    
    # Print the content of the final message
    if hasattr(final_message, 'content'):
        print(final_message.content)
    
    # # If there were tool calls, you can inspect them here if needed
    # print(final_message)
    # if hasattr(final_message, "tool_call_id"):
    #     print("\nTools used:", final_message.name)
    # else:
    #     print("\nTools used:", final_message.tool_calls)
    
    print(f"\n Message Type: {final_message.type}")
    print(f"\tFull Message: {final_message.content}")
    
    # Only check for tool_calls if it's an AIMessage
    if tool_calls:=getattr(final_message, "tool_calls", None):
        print("\n Tools used:")
        for tc in tool_calls:
            print(f"   - {tc.get('name', 'unknown')}: {tc.get('args', {})}")

if __name__ == "__main__":
    # Check for API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment variables.")
        sys.exit(1)
        
    args = parser.parse_args()
    if args.query:
        run_cli(args)
    else:
        mcp_server.run(transport="http", host="127.0.0.1", port=8002, path="/mcp")