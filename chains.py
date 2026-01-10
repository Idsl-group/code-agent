import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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