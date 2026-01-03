import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. System Prompt: Defines the Persona and Constraints
SYSTEM_PROMPT = """
You are a deterministic TOOL-CALLING AGENT.

Your sole responsibility is to select and call exactly ONE appropriate tool
to fulfill the user's request.

STRICT RULES (non-negotiable):
- You MUST respond with a tool call.
- You MUST NOT produce natural language explanations.
- You MUST NOT answer directly.
- You MUST NOT ask clarifying questions.
- You MUST NOT output markdown, code blocks, or commentary.

Tool usage rules:
- Select the single most appropriate tool from the list below.
- The tool name MUST match exactly.
- The arguments MUST strictly follow the tool's schema.
- All required fields MUST be provided.
- Do NOT invent tools or arguments.

Failure handling:
- If no tool applies, call the tool named "no_op" with an empty argument object.
- Never respond with free text under any circumstance.

Code generation rules (when applicable):
- Generated code MUST be valid and executable.
- Follow PEP-8 conventions.
- Include concise, meaningful comments.
- Do NOT explain the code outside the tool call.

Available tools:
{tools}
"""

# 2. Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{chat_messages}") # This injects the conversation history
])

# 3. LLM Initialization
def get_llm(api_key=None):
    """Initialize the model. Assumes OPENAI_API_KEY is set in env."""
    assert api_key
    
    base_url = None
    if api_key=="OLLAMA":
        base_url = os.getenv("OLLAMA_SERVER")
    assert base_url
    
    return ChatOpenAI(
        base_url = base_url, 
        model = os.getenv("MODEL_ID"),
        api_key = api_key, 
        temperature = 0.4,
    )

def get_coding_chain(api_key=None):
    """Combines prompt and LLM."""
    return (prompt | get_llm(api_key))