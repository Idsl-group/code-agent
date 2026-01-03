import json
import re
import uuid
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL
)

def strip_json_markdown(cleaned: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` markdown fences from a string.
    """
    cleaned = cleaned.strip()

    # Remove opening ```json or ```
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)

    # Removeopening and closing ```
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    return cleaned

def _extract_tool_payload(llm, tools, message: AIMessage) -> Optional[Dict[str, Any]]:
    """
    Extract and validate a tool-call payload from an AIMessage.

    The function:
    1. Extracts raw JSON text from <tool_call>...</tool_call>
    2. Uses the LLM to normalize / repair the JSON if needed
    3. Parses the result with json.loads safely
    4. Returns a dict with keys: {"name": str, "args": dict}

    Returns None if no tool call is present.
    Raises ValueError if validation fails.
    """
    if not isinstance(message, AIMessage) or not message.content:
        return None

    match = _TOOL_CALL_RE.search(message.content)
    if not match:
        return None

    raw = match.group(1).strip()

    # Step 1: Ask the LLM to validate / repair JSON
    validator_prompt = (
    "You are a STRICT JSON TOOL-CALL VALIDATOR and REPAIRER.\n\n"
    "GOAL:\n"
    "Return exactly ONE JSON object representing a tool call.\n"
    "The output MUST be syntactically valid JSON and MUST conform to the tool schema.\n\n"
    "STRICT OUTPUT RULES (NON-NEGOTIABLE):\n"
    "- Output ONLY valid JSON, no markdown, no code fences, no explanations.\n"
    "- The JSON must be a single object (not a list).\n"
    "- Do NOT add extra top-level keys.\n"
    "- Do NOT rename keys.\n"
    "- Do NOT change the selected tool name.\n"
    "- Ensure ALL required tool arguments are present under `arguments`.\n"
    "- If the input cannot be repaired into schema-valid JSON, output EXACTLY: null\n\n"
    "CRITICAL JSON ESCAPING RULES (MUST FOLLOW):\n"
    "- Every string value MUST be valid JSON string syntax.\n\n"
    "###Escape all special characters inside code values:\n"
    "- Newlines must be written as \\n (not literal newlines)\n"
    "- Tabs must be written as \\t\n"
    "- Carriage returns must be written as \\r\n"
    "- Backslashes must be escaped as \\\\\n"
    "- Double quotes inside strings must be escaped as \\\"\n"
    "- Do NOT output unescaped control characters (U+0000 through U+001F) in any string.\n"
    "- If a value contains code, keep it as a single JSON string with proper escapes.\n\n"
    "REQUIRED TOOL-CALL SHAPE:\n"
    '{"name": "<tool_name>", "arguments": { <tool_argument_key_values> } }\n\n'
    "AVAILABLE TOOL SCHEMAS (authoritative):\n"
    f"{tools}\n\n"
    "INPUT (may be invalid JSON):\n"
    f"{raw}\n"
)

    llm_response = llm.invoke([
        HumanMessage(content=validator_prompt)
    ])

    cleaned = llm_response.content.strip()

    # Step 2: Parse safely with json.loads
    try:
        print(cleaned)
        payload = json.loads(strip_json_markdown(cleaned))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM failed to produce valid JSON.\n"
            f"Original:\n{raw}\n\n"
            f"LLM output:\n{cleaned}"
        ) from e

    # Step 3: Validate expected schema
    name = payload.get("name")
    args = payload.get("arguments")

    if not isinstance(name, str):
        raise ValueError(f"Invalid or missing tool name: {payload}")

    if args is None:
        args = {}
    elif isinstance(args, str):
        # Handle double-encoded JSON
        try:
            args = json.loads(args)
        except json.JSONDecodeError as e:
            raise ValueError(f"Arguments string is not valid JSON: {args}") from e

    if not isinstance(args, dict):
        raise ValueError(f"Tool arguments must be a dict: {args}")

    return {
        "name": name,
        "args": args,
    }


def _to_tool_call_ai_message(llm, tools, message: AIMessage) -> AIMessage:
    """
    Convert a text-based tool call inside AIMessage.content into an AIMessage
    that LangGraph ToolNode can execute (via message.tool_calls).
    """
    parsed = _extract_tool_payload(llm, tools, message)
    if parsed is None:
        # No tool call found, return unchanged
        return message

    tool_call_id = f"call_{uuid.uuid4().hex}"

    # OpenAI-style tool_calls structure used by LangGraph ToolNode
    tool_calls = [{
        "id": tool_call_id,
        "name": parsed["name"],
        "args": parsed["args"],
        "type": "tool_call",
    }]

    # Keep original content optional, but it's safer to blank it out
    # so downstream doesn't treat it as a "final answer".
    tool_message = AIMessage(
        content="",
        tool_calls=tool_calls,
        additional_kwargs=message.additional_kwargs,
        response_metadata=message.response_metadata,
        id=message.id,
        usage_metadata=message.usage_metadata,
    )
    
    return tool_message


# This is the chain you plug in between the model and ToolNode
tool_call_normalizer = RunnableLambda(_to_tool_call_ai_message)

# Example composition:
# chain = prompt | llm | tool_call_normalizer
