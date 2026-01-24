from typing import Annotated, List, Sequence, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    The state of the agent.
    
    'messages' is a list of chat messages. 
    The `add_messages` annotation ensures that new messages are appended 
    rather than overwriting the history.
    """
    messages: Annotated[Sequence, add_messages]
    reflections: Annotated[Sequence, add_messages]
    thoughts: Annotated[Sequence, add_messages]
    actions: Annotated[Sequence, add_messages]
    reflection_ids: List[int] = []
    thought_ids: List[int] = []
    action_ids: List[int] = []