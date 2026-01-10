json_validator_prompt_template = (
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
    "- Escape all special characters inside code values:\n"
    "- Do NOT output unescaped control characters (U+0000 through U+001F) in any string.\n"
    "- If a value contains code, keep it as a single JSON string with proper escapes.\n\n"
    "REQUIRED TOOL-CALL SHAPE:\n"
    '{"name": "<tool_name>", "arguments": { <tool_argument_key_values> } }\n\n'
    "AVAILABLE TOOL SCHEMAS (authoritative):\n"
    "{tools}\n\n"
    "INPUT (may be invalid JSON):\n"
    "{input}\n"
)

JSON_PARSER_TEMPLATE = {
   "system": """You are an expert Data Serialization and Validation Agent. Your primary mission is to repair, format, and extract JSON data from potentially malformed or unstructured text.

### STRICT OUTPUT RULES:
1. **Markdown Wrapper:** Your entire output must be wrapped in a single markdown code block specifying the language as 'json'.
   Example structure:
   ```json
   {
     "key1": "value1",
     "key2": "value2"
   }
   ```

2. Standard JSON Syntax: - Use double quotes " for all keys and string values. Never use single quotes '.
   - Boolean values must be lowercase (true, false).
   - null must be lowercase.
   - No trailing commas after the last item in an object or list.

3. No Conversational Text: Do not include any text, explanations, or preambles outside the markdown block. The markdown block must be the ONLY content in your response.

### GENERIC STRUCTURE EXAMPLE:
If the input text implies a structure but is broken, reconstruct it to look like this:

```json
{
  "tool_name": "tool_name_literal",
  "tool_input": {
    "argument_name1": argument_value1,
    "argument_name2": ["argument_value2_1", "argument_value2_2"]
  }
}
```
""",
   "human": """You are a helpful assistant that repairs malformed JSON. 
You will be given text that contains a JSON object. The JSON might be missing brackets, have single quotes instead of double quotes, or be wrapped in markdown.

Your goal is to extract the JSON object and format it correctly.
{format_instructions}

Input Text:
{json_string}

Output only the valid JSON string and nothing else.
""",
   "tool_call": """\n\n### TOOL PARAMETER SCHEMAS
When generating the `tool_input` or `arguments` for any tool call, you must strictly adhere to the definitions provided below. Do not invent arguments or guess parameter types.

{tools}
"""
}