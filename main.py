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

parser = argparse.ArgumentParser()
parser.add_argument("--query", required=True, type=str, help="Eg: Write a python script that calculates fibonacci numbers.")

def run_cli(args):
    # 1. Get user query from Args
    user_query = args.query

    # 2. Initialize the State
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_query)]
    }

    # 3. Build the Graph
    app = build_graph(os.getenv("OPENAI_API_KEY"))

    print(f"\n Agent Processing: '{user_query}'\n")
    print("-" * 50)

    # 4. Run the Graph
    # stream() prints updates as they happen; invoke() waits for the end.
    # Using invoke for simpler final output in this script.
    final_state = app.invoke(initial_state)

    # 5. Output the final result
    final_message = final_state["messages"][-1]
    
    print("-" * 50)
    print("\n Agent Output:\n")
    
    # Print the content of the final message
    if hasattr(final_message, 'content'):
        print(final_message.content)
    
    # If there were tool calls, you can inspect them here if needed
    print(final_message)
    if hasattr(final_message, "tool_call_id"):
        print("\nTools used:", final_message.name)
    else:
        print("\nTools used:", final_message.tool_calls)

if __name__ == "__main__":
    # Check for API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment variables.")
        sys.exit(1)
        
    args = parser.parse_args()
    run_cli(args)