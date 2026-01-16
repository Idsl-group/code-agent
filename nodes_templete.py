import os
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypeVar, Union, Literal, Callable

class ReflectionOutput(BaseModel):
    decision: Literal["DONE", "CONTINUE", "USER_INPUT"] = Field(
        "CONTINUE", description="Whether the original task is complete, needs another iteration, or requires an input from the human user."
    )
    instruction: Optional[str] = Field(
        None,
        description="A detailed instruction text for the selected decision."
    )

REFLECTION_OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "instruction"],
    "properties": {
        "decision": {"type": "string", "enum": ["DONE", "CONTINUE", "USER_INPUT"]},
        "instruction": {"type": "string"},
    }
}

REFLECTION_OUTPUT_JSON = """```json
{
  // REQUIRED
  "decision": "<string: one of 'DONE' | 'CONTINUE' | 'USER_INPUT'>",
  // REQUIRED
  "instruction": "<string: A detailed instruction text for the selected decision.>"
}
```
"""

REFLECTION_SYSTEM_PROMPT = """You are a deterministic REFLECTION AGENT.

You evaluate whether the tool execution result satisfies the original user task.

You MUST output exactly ONE valid JSON object that conforms to the provided JSON schema.
No markdown, no code fences, no explanations, no extra text.

Do not execute tools.
Do not generate new artifacts.
Return only the decision and, if needed, a fix instruction.
"""

REFLECTION_TEMPLATE_PROMPT = """You are a REFLECTION AGENT that reviews tool execution results.

Your job is to output a JSON object in markdown format that conforms EXACTLY to the given schema.

Available Tools:
{tools}

Original user task:
{original_request}

Current Tool execution result:
TOOL_NAME: {tool_name}
TOOL_OUTPUT: {tool_result}

DECISION RULES:
- decision = "DONE" if the tool execution result fully satisfies the original user task with no errors, missing requirements, or follow-up needed.
- decision = "CONTINUE" if any part of the user task is incomplete, incorrect, unusable, or requires revision, and the agent can proceed without asking the user anything.
- decision = "USER_INPUT" if the task cannot be completed without additional information from the user (for example: missing file path, missing required parameters, ambiguity between multiple valid options, access/permission constraints, or the tool output indicates a recoverable issue that requires the user to decide).

INSTRUCTION FIELD RULES:
- If decision = "DONE", include "instruction" describing why the original user task is complete based on the conversation history and the current Tool execution result.
- If decision = "CONTINUE", include "instruction" describing the fix and the tool call to be executed in sufficient detail for the next agent step.
- If decision = "USER_INPUT", include "instruction" as a single, direct sentence telling exactly what information the user must provide, do NOT include multiple questions.

OUTPUT RULES (STRICT):
- Output ONLY valid JSON.
- No markdown, no code fences, no commentary.
- Do not include any keys other than those allowed by the schema.

JSON SCHEMA:
{reflection_schema}
"""

USER_INPUT_REFLECTION_DIRECTIVE = """The following input was received from the user:

<USER_INPUT>
{user_input}
</USER_INPUT>

- Determine whether this input is sufficient to complete the original task.
- If the task can be completed without any additional user information, follow the normal decision rules and execute the next decision.
- If the task cannot stiil be completed without more information from the user, \
    include an instruction that clearly and concisely states exactly what additional information the user must provide next.
- Do not output decision `DONE` if steps still need to be performed to complete the original user task, \
    `CONTINUE` with the next appropriate decision and instruction.
"""