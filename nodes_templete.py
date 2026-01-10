import os
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypeVar, Union, Literal, Callable

class ReflectionOutput(BaseModel):
    decision: Literal["DONE", "CONTINUE", "USER_INPUT"] = Field(
        ..., description="Whether the task is complete or needs another iteration, or requires an input from the human user."
    )
    instruction: Optional[str] = Field(
        None,
        description="A detailed fix instruction, required if decision is CONTINUE or USER_INPUT."
    )

REFLECTION_OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["decision"],
    "properties": {
        "decision": {"type": "string", "enum": ["DONE", "CONTINUE", "USER_INPUT"]},
        "instruction": {"type": "string", "minLength": 1, "maxLength": os.getenv("REFLECTION_MAX_TOKENS", 1024)},
    },
    "allOf": [
        {
            "if": {"properties": {"decision": {"const": "CONTINUE"}}},
            "then": {"required": ["instruction"]},
        },
        {
            "if": {"properties": {"decision": {"const": "DONE"}}},
            "then": {"not": {"required": ["instruction"]}},
        },
    ],
}

REFLECTION_SYSTEM_PROMPT = """You are a deterministic REFLECTION AGENT.

You evaluate whether the tool execution result satisfies the original user request.

You MUST output exactly ONE valid JSON object that conforms to the provided JSON schema.
No markdown, no code fences, no explanations, no extra text.

Do not execute tools.
Do not generate new artifacts.
Return only the decision and, if needed, a fix instruction.
"""

REFLECTION_TEMPLATE_PROMPT = """You are a REFLECTION AGENT that reviews tool execution results.

Your job is to output ONE JSON object that conforms EXACTLY to the schema.

DECISION RULES:
- decision = "DONE" if the tool output fully satisfies the original request with no errors, missing requirements, or follow-up needed.
- decision = "CONTINUE" if any part of the request is incomplete, incorrect, unusable, or requires revision, and the agent can proceed without asking the user anything.
- decision = "USER_INPUT" if the task cannot be completed without additional information from the user (for example: missing file path, missing required parameters, ambiguity between multiple valid options, access/permission constraints, or the tool output indicates a recoverable issue that requires the user to decide).

INSTRUCTION FIELD RULES:
- If decision = "CONTINUE", include "instruction" describing the fix and the tool call to be executed in sufficient detail for the next agent step.
- If decision = "DONE", do NOT include "instruction".
- If decision = "USER_INPUT", include "instruction" as a single, direct sentence telling exactly what information the user must provide, do NOT include multiple questions.

OUTPUT RULES (STRICT):
- Output ONLY valid JSON.
- No markdown, no code fences, no commentary.
- Do not include any keys other than those allowed by the schema.

JSON SCHEMA:
{reflection_schema}

Available Tools:
{tools}

Original user request:
{original_request}

Current Tool execution result:
TOOL_NAME: {tool_name}
TOOL_OUTPUT: {tool_result}
"""