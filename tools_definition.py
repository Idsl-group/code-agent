import os, sys
import json
from typing import Literal
from os.path import isfile, isdir, join, abspath

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain.tools.render import render_text_description_and_args
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, AnyMessage
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from chains import get_llm
from schemas import AgentState
from tool_formatter import json_output_parser

# --- Tools Definition ---
# These are the capabilities the Agent has access to.

ROOT_DIR = os.getenv("ROOT_DIR", "./")
API_KEY = os.getenv("OPENAI_API_KEY")
TOOL_DEBUG = True if int(os.getenv("TOOL_DEBUG", 0)) else False
FILE_DATA = {}

def set_root_dir(root_dir):
    global ROOT_DIR
    ROOT_DIR = root_dir
    try:
        assert isdir(ROOT_DIR)
    except:
        print(f"ROOT DIR `{root_dir}` does not exist")
        ROOT_DIR = "./"
    return ROOT_DIR

@tool("conversational_response")
def conversational_response(response: str) -> str:
    """
    Return a non-terminal, conversational message to the user.

    PURPOSE:
        This tool is used to send an intermediate or conversational response
        to the user that does NOT represent the final completion of the task.

        It should be used for:
        - Clarifications
        - Status updates
        - Explanatory messages
        - Requests for additional information
        - Intermediate reasoning summaries

        This tool does NOT end the workflow.

    WHEN TO USE:
        Call this tool if the user interaction requires:
        - Ongoing dialogue
        - Additional steps before a final answer
        - Acknowledgement of input before proceeding
        - Explanations without task completion

    WHEN NOT TO USE:
        - If the task is fully complete
        - If the user expects a final, definitive answer
        - If no further interaction is required

    PARAMETERS:
        response (str):
            The conversational message to return to the user.
            This message should be:
            - Clear and concise
            - Helpful and context-aware
            - Free of internal reasoning or system details

    RETURNS:
        str:
            The provided conversational response, returned verbatim.

    IMPORTANT RULES FOR TOOL-CALLING MODELS:
        - Do NOT include tool names or internal logic in the response
        - Do NOT mark the task as complete
        - Do NOT provide a final answer using this tool
        - Always use this tool instead of free-text replies for intermediate communication

    EXAMPLE TOOL CALL (JSON):
        {
            "response": "I've located the file and will update it next."
        }
    """
    
    if TOOL_DEBUG:
        print("INSIDE conversational_response")
    return response

@tool("final_answer")
def final_answer(answer: str) -> str:
    """
    Return the final, user-facing answer and terminate the workflow.

    PURPOSE:
        This tool is used to deliver the final result of the task to the user.
        Calling this tool indicates that:
        - All required steps have been completed
        - No further tools should be called
        - The workflow should terminate

        This is the ONLY tool that represents task completion.

    WHEN TO USE:
        Call this tool if:
        - The user's request has been fully and correctly satisfied
        - No further clarification or iteration is needed
        - The final output is ready to be presented

    WHEN NOT TO USE:
        - If additional tools must still be executed
        - If the task is incomplete or partially complete
        - If the user expects further interaction or refinement

    PARAMETERS:
        answer (str):
            The final response to the user.
            This should be:
            - Complete and accurate
            - Clearly formatted
            - Free of internal reasoning, system messages, or tool metadata

    RETURNS:
        str:
            The provided final answer, returned verbatim.

    IMPORTANT RULES FOR TOOL-CALLING MODELS:
        - This tool MUST be called exactly once per completed task
        - Do NOT include analysis or internal reasoning
        - Do NOT call any other tools after this one
        - Do NOT provide the final answer outside this tool

    EXAMPLE TOOL CALL (JSON):
        {
            "answer": "The Fibonacci function has been written to fibonacci.py successfully."
        }
    """
    
    if TOOL_DEBUG:
        print("INSIDE final_answer")
    return answer

@tool("read_file")
def read_file(file_path: str) -> str:
    """
    Read and return the full contents of an existing file from the local filesystem.

    PURPOSE:
        This tool is used to retrieve and inspect the contents of an existing file.
        It is strictly READ-ONLY and must never be used to modify, create, or delete files.

        This tool should be called whenever the user needs to:
        - View the contents of a file
        - Inspect existing code or text
        - Read configuration files or scripts
        - Understand what is currently stored in a specific file

    WHEN TO USE:
        Call this tool if the user explicitly or implicitly requests to:
        - "read <file>"
        - "open <file>"
        - "show the contents of <file>"
        - "display <file>"
        - "inspect <file>"
        - "what is inside <file>"

    WHEN NOT TO USE:
        - If the user asks to write, update, or overwrite a file
        - If the user asks to create a new file
        - If the user requests an explanation without needing file contents
        - If the file path does not exist
        - If the user requests directory listings or metadata

    PARAMETERS:
        file_path (str):
            A relative file path from ROOT_DIR pointing to an EXISTING file.
            The file must already exist on disk.

            Examples:
                "main.py"
                "src/utils/helpers.py"
                "config/settings.yaml"

    BEHAVIOR:
        - Resolves the absolute file path relative to ROOT_DIR
        - Verifies that ROOT_DIR exists
        - Verifies that the target file exists
        - Opens the file in read-only mode
        - Reads and returns the entire file contents as a string

    RETURNS:
        str:
            On success:
                A string containing the full contents of the file,
                prefixed with a short label identifying the file name.

            On failure:
                A descriptive error message indicating that the file
                could not be found or read.

    FAILURE CONDITIONS:
        - ROOT_DIR does not exist
        - file_path does not refer to an existing file
        - Insufficient permissions to read the file
        - Any I/O error during file access

    IMPORTANT RULES FOR TOOL-CALLING MODELS:
        - You MUST provide the `file_path` argument
        - Do NOT invent or guess file paths
        - Do NOT include explanations outside the tool call
        - Do NOT attempt to modify file contents
        - Do NOT call this tool unless file contents are explicitly required

    EXAMPLE TOOL CALL (JSON):
        {
            "file_path": "main.py"
        }
    """
    
    if TOOL_DEBUG:
        print("INSIDE read_file")
    fpath = join(ROOT_DIR, file_path)
    fpath = abspath(fpath)
    try:
        assert isdir(ROOT_DIR)
        assert isfile(fpath)
        with open(fpath, 'r') as f:
            return f"Contents of file `{file_path}`: \n\n<FILE_CONTENT>\n{f.read()}\n</FILE_CONTENT>"
    except Exception as e:
        return f"Error {e}: File '{file_path}' not found."
    
@tool("generate_text_file")
def generate_text_file(content: str) -> str:
    """_summary_

    Args:
        content (str): _description_

    Returns:
        str: _description_
    """
    
    global FILE_DATA
    if TOOL_DEBUG:
        print("INSIDE generate_text_file")
    f_name = None
    if "generate_text_file" not in FILE_DATA:
        FILE_DATA["generate_text_file"] = {}
    
    fname_prompt_template = ChatPromptTemplate.from_messages([
        ("system", ),
        ("humnan", )
    ])
    
    fina_name_llm = get_llm(API_KEY)
    file_name_chain = (fname_prompt_template | fina_name_llm)
    file_name_response = file_name_chain.invode({
        "file_content": content,
        "invalid_filenames": "",
    })
    file_name_dict = json_output_parser(fina_name_llm, file_name_response.content, keys=["file_name"])
    f_name = file_name_dict["file_name"]
    
    FILE_DATA["generate_text_file"][f_name] = content
    
    assert f_name
    return f"Text file contents have been generated and saved as `{f_name}` in the file database. Use the `write_file` tool to save it to the local disk."

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
            The full text/code content to write into the file.
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
            "content": "def main():\\n    print('Hello, world!')\\n\\nif __name__ == '__main__':\\n    main()\\n"
        }
    """
    if TOOL_DEBUG:
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
    "read_file": read_file,
    "write_file": write_file,
    "conversational_response": conversational_response,
    "final_answer": final_answer,
}
