import os, sys
from os.path import isfile, isdir, join, abspath

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain.tools.render import render_text_description_and_args
from chains import get_llm, prompt
from schemas import AgentState

from tool_formatter import _to_tool_call_ai_message

# --- Tools Definition ---
# These are the capabilities the Agent has access to.

ROOT_DIR = os.getenv("ROOT_DIR")
API_KEY = None

def set_api_key(api_key):
    global API_KEY
    API_KEY = api_key
    return API_KEY

@tool("read_file")
def read_file(file_path: str) -> str:
    """Tool to read the contents of a file from the local filesystem."""
    
    print("INSIDE read_file")
    fpath = join(ROOT_DIR, file_path)
    fpath = abspath(fpath)
    try:
        assert isdir(ROOT_DIR)
        assert isfile(fpath)
        with open(fpath, 'r') as f:
            return f"Contemts of file `{file_path}`: \n\n{f.read()}"
    except Exception as e:
        return f"Error {e}: File '{file_path}' not found."

@tool("write_file")
def write_file(file_path: str, content: str) -> str:
    """
    Write text content to an existing file on the local filesystem.

    PURPOSE:
        This tool MUST be used whenever the user asks to:
        - Write code to a file
        - Save generated code, scripts, or configuration to disk
        - Overwrite the contents of an existing file
        - Persist generated output to a specific file path

        The tool overwrites the target file completely.
        It does NOT create new files or directories.

    WHEN TO USE:
        Call this tool if the user explicitly or implicitly requests:
        - "write this to <file>"
        - "save the code in <file>"
        - "update <file> with the following content"
        - "store the output in <file>"

    WHEN NOT TO USE:
        - If the user only asks for an explanation
        - If the user asks to read or inspect a file
        - If the file path does not already exist
        - If no file output is requested

    PARAMETERS:
        file_path (str):
            A relative path (from ROOT_DIR) to an EXISTING file.
            The file MUST already exist.
            Example:
                "main.py"
                "src/utils/helpers.py"

        content (str):
            The full text content to write into the file.
            This content completely replaces the existing file contents.
            For code generation:
                - Must be valid, executable code
                - Must follow PEP 8 style guidelines
                - Must include appropriate inline comments

    BEHAVIOR:
        - Resolves the absolute path using ROOT_DIR
        - Verifies that ROOT_DIR exists
        - Verifies that the target file exists
        - Overwrites the file with the provided content
        - Returns a success message including the written content

    RETURNS:
        str:
            On success:
                A confirmation message indicating the file path
                and echoing the written content.

            On failure:
                A descriptive error message explaining what went wrong.

    FAILURE CONDITIONS:
        - ROOT_DIR does not exist
        - file_path does not point to an existing file
        - Insufficient filesystem permissions
        - Any I/O error during writing

    IMPORTANT RULES FOR TOOL-CALLING MODELS:
        - You MUST provide BOTH `file_path` and `content`
        - Do NOT omit required arguments
        - Do NOT include explanations outside the tool call
        - Do NOT attempt to create new files
        - Do NOT call this tool unless a file write is explicitly required

    EXAMPLE TOOL CALL (JSON):
        {
            "file_path": "main.py",
            "content": "def main():\n    print('Hello, world!')\n\nif __name__ == '__main__':\n    main()\n"
        }
    """
    
    print("INSIDE write_file")
    print(file_path)
    print(content)
    
    fpath = join(ROOT_DIR, file_path)
    fpath = abspath(fpath)
    try:
        assert isdir(ROOT_DIR)
        with open(fpath, 'w') as f:
            f.write(content)
        assert isfile(fpath)
        return f"Successfully wrote to '{file_path}'.\n\nPYTHON CODE:\n\n{content}"
    except Exception as e:
        return f"Error writing file `{file_path}`: {e}"

# List of available tools
tools = {
    # "read_file": read_file,
    "write_file": write_file,
}

# --- Node Functions ---

def tool_calling_node(state: AgentState):
    print("INSIDE TOOL CALLING NODE")
    # Bind tools to the LLM so it knows they exist
    llm_with_tools = get_llm(API_KEY).bind_tools(list(tools.values()))
    
    tools_str = render_text_description_and_args(list(tools.values()))
    
    tool_prompt = prompt.partial(tools=tools_str)
    chain = (tool_prompt | llm_with_tools)
    
    response = chain.invoke({"chat_messages": state["messages"]})
    print(response)
    if not hasattr(response, "tool_call_id"):
        print("USING TOOL WRAPPER")
        response = _to_tool_call_ai_message(get_llm(API_KEY), tools_str, response)
    return {"messages": [response]}

# We use LangGraph's built-in ToolNode for executing tool calls
tool_node = ToolNode(list(tools.values()))