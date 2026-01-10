import json, ast
import re, copy
import uuid
from typing import Any, Dict, Optional, Literal

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.chat import ChatPromptTemplate

from tools_template import *

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL
)

def json_output_parser(llm, message_content, required_keys=None, force_parse=False, tools=None, max_retries=5):
    """
    Parses a JSON string from LLM output, with an LLM-based fallback for repair.
    """
    
    def extract_json_str(text):
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()

    def local_parse(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return None

    try:
        json_str = extract_json_str(message_content)
        parsed_json = local_parse(json_str)
        if parsed_json:
            if required_keys is not None:
                assert set(list(parsed_json.keys()))==set(required_keys)
                return {k: parsed_json.get(k) for k in required_keys if k in parsed_json}
            return parsed_json
    except Exception as e:
        if force_parse:
            raise ValueError(f"Failed to parse JSON due to error `{e}` \n{message_content}")
        print(f"Using LLM JSON PARSER due to error: {e}")

    current_text = copy.deepcopy(message_content)
    tool_str = JSON_PARSER_TEMPLATE["tool_call"].format(tools=tools) if tools is not None else ""
    if required_keys:
        format_instr = f"Ensure the JSON contains the following keys: {', '.join(required_keys)}.{tool_str}"
    else:
        format_instr = "Ensure the output is a valid JSON.{tool_str}"

    repair_chain = (
        ChatPromptTemplate.from_messages([
            ("system", JSON_PARSER_TEMPLATE["system"]),
            ("human", JSON_PARSER_TEMPLATE["human"])
        ]) 
        | llm 
        | StrOutputParser()
    )

    for _ in range(max_retries):
        try:
            fixed_str = repair_chain.invoke({
                "json_string": current_text,
                "format_instructions": format_instr
            })
            clean_fixed_str = extract_json_str(fixed_str)
            parsed_json = local_parse(clean_fixed_str)
            
            if parsed_json:
                if required_keys:
                    missing = [k for k in required_keys if k not in parsed_json]
                    if len(missing)==0:
                        return {k: parsed_json[k] for k in required_keys}
                else:
                    return parsed_json
                    
            current_text = fixed_str
        except Exception:
            continue
    raise ValueError(f'Failed to parse JSON after {max_retries} attempts. The following is the input JSON data:\n{message_content}\n\
Analyse the changes necessary and revise the $JSON_BLOB to re-initiate the "Thought-Action-Observation" step for the required task in the workflow. \
Ensure that your response aligns with the specified $JSON_BLOB format.')

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
    json_validator_prompt = json_validator_prompt_template.format(tools=tools, input=raw)
    llm_response = llm.invoke([
        HumanMessage(content=json_validator_prompt)
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