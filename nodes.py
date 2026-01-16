import os, sys
import json, uuid
from os.path import isfile, isdir, join, abspath
from argparse import Namespace
from typing  import List, Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain.tools.render import render_text_description_and_args
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, ToolCall, AnyMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from chains import get_llm
from schemas import AgentState

from chain_templete import TOOL_SYSTEM_PROMPT, TOOL_HUMAN_PROMPT
from tools_definition import set_root_dir, tools
from tool_formatter import _to_tool_call_ai_message, json_output_parser
from nodes_templete import *

API_KEY = None
MAX_RETRIES = 5

def set_api_key(api_key):
    global API_KEY
    API_KEY = api_key
    return API_KEY

class ToolCallingAgent:
    def __init__(self, api_key, tool_list: List[Any]=None):
        if tool_list:
            self.tool_node = ToolNode(tool_list)
        else:
            self.tool_node = ToolNode(list(tools.values()))
        self.api_key = api_key
        self.latest_decision = None
        self.latest_instruction = None
        self.reflection_directive = None
        
    def should_continue(self, state: AgentState) -> Literal["tool_node", "reflect", "__end__"]:
        """
        Determines the next step based on the Agent's last message.
        If the Agent made a tool call, we route to 'tools'.
        Otherwise, we finish (END).
        """
        print("\n" + "="*30)
        print("ROUTING: should_continue_after_tool_call")
        print("="*30)
        
        messages = state["messages"]
        print("STATE MESSAGES:")
        print([f"{f.type}:{f.content}" for f in messages])
        last_message = messages[-1]
        print(f" Total messages: {len(messages)}")
        print(f" Last message is tool call: {hasattr(last_message, 'tool_calls')}")
        
        try:
            tool_calls = getattr(last_message, 'tool_calls', None)
            if tool_calls[0]["name"] == "final_answer":
                return  "__end__"
            elif tool_calls:=getattr(last_message, 'tool_calls', None):
                print(f"\tFound tool call(s):\n{tool_calls}")
                print("\tROUTING TO: tools")
                print("="*30 + "\n")
                return "tool_node"
            else:
                raise ValueError("No valid tool_calls found in agent state.")
        except Exception as e:
            # If no tool calls, we are done
            print("No tool calls found")
            print("ROUTING TO: __end__")
            print("="*30 + "\n")
            return "reflect"

    def tool_calling_node(self, state: AgentState):
        global API_KEY
        print("INSIDE TOOL CALLING NODE")
        
        if state.get("reflections"):
            for r in reversed(state["reflections"]):
                self.latest_decision = r.additional_kwargs.get("decision", None)
                self.latest_instruction = r.additional_kwargs.get("instruction", None)
                break
        
        tool_list = list(tools.values())
        # Bind tools to the LLM so it knows they exist
        tools_str = render_text_description_and_args(tool_list)
        llm_with_tools = get_llm(API_KEY).bind_tools(tool_list)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", TOOL_SYSTEM_PROMPT),
            ("human", TOOL_HUMAN_PROMPT),
            MessagesPlaceholder(variable_name="chat_messages", optional=True), # This injects the conversation history
        ])
        tool_prompt = prompt.partial(tools=tools_str)
        chain = (tool_prompt | llm_with_tools)
        
        chat_messages = state["messages"][1:] if len(state["messages"]) > 1 else []
        response = chain.invoke({
            "user_input": state["messages"][0].content,
            "chat_messages": chat_messages,
            })
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            print("USING TOOL WRAPPER")
            response = _to_tool_call_ai_message(get_llm(API_KEY), tools_str, response)
        elif len(tool_calls)>1:
            print("ENFORCING SINGLE TOOL")
            response.tool_calls = [tool_calls[0]]
            
        print(response)
        return {"messages": [response]}

class ReflectionAgent:
    """
    Agent responsible for reflecting on tool execution results
    and deciding whether the task is DONE or should CONTINUE.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.reflection_state = None
        print(f"\tReflectionAgent initialized")
        
    def user_input_node(self, state: AgentState):
        user_inp = ""
        while True:
            query = state["reflections"][-1].additional_kwargs["instruction"]+"\n\nPlease enter a prompt to CONTINUE...\n\n"
            user_inp = input(query)
            if len(user_inp.strip())>0:
                break
        tool_id = str(uuid.uuid4())
        tool_calls = {"name": "user_input", "args": {"query": query, "input": user_inp}}
        state["messages"].extend([
            AIMessage(content=f"Executing tool-call for user input: \n\n```json\n{json.dumps(tool_calls, indent=2)}\n```\n"),
            ToolMessage(name="user_input", content=user_inp, tool_call_id=tool_id),
        ])
        return state
    
    def reflect(self, state: AgentState):
        print("\n" + "="*60)
        print("NODE: properties_agent.reflect")
        print("="*60)
        
        messages = state["messages"]
        original_request = messages[0].content
        assert len(original_request.strip())>0
        print("REFLECTION MESSAGES:")
        print([f"{f.type}:{f.content}" for f in messages])
        print(f"Original Request: {original_request}...")
        
        last_message = messages[-1]
        tool_name = getattr(last_message, "name", None)
        tool_result = getattr(last_message, "content", None)
        assert tool_name
        assert tool_result
        print(f"Tool Name: {tool_name} | Tool Result Preview: {tool_result}\n")
        
        if tool_name in ("user_input"):
            tool_result = USER_INPUT_REFLECTION_DIRECTIVE.format(user_input=tool_result)
    
        tools_str = render_text_description_and_args(list(tools.values()))
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_messages", optional=True),
            HumanMessage(content=REFLECTION_TEMPLATE_PROMPT),
        ])
        
        print(" Invoking LLM for reflection...")
        llm = get_llm(self.api_key)
        try:
            reflection_llm = llm.with_structured_output(ReflectionOutput)
            reflection_chain = (prompt_template | reflection_llm)
        except Exception as e:
            print("Error binding SCHEMA to LLM.")
            raise ValueError(str(e))
        
        chat_messages = messages[:-1]
        response = None
        retry_ctr = 0
        while retry_ctr<MAX_RETRIES:
            try:
                response = reflection_chain.invoke({
                    "chat_messages": chat_messages,
                    # "reflection_schema": json.dumps(REFLECTION_OUTPUT_SCHEMA),
                    "reflection_schema": REFLECTION_OUTPUT_JSON,
                    "tools": tools_str,
                    "tool_name": tool_name,
                    "tool_result": tool_result,
                    "original_request": original_request
                })
                break
            except Exception as e:
                retry_ctr += 1
                print(f"Structured output failed with error: {e} \nRetrying with `json_output_parser`")
                reflection_json_chain = (prompt_template | llm)
                ai_response = reflection_json_chain.invoke({
                    "chat_messages": chat_messages,
                    # "reflection_schema": json.dumps(REFLECTION_OUTPUT_SCHEMA),
                    "reflection_schema": REFLECTION_OUTPUT_JSON,
                    "tools": tools_str,
                    "tool_name": tool_name,
                    "tool_result": tool_result,
                    "original_request": original_request
                })
                response = json_output_parser(llm, ai_response.content, required_keys=["decision", "instruction"], force_parse=False)
                if not isinstance(response, dict):
                    response = None
                    continue
            finally:
                print(response)
                if isinstance(response, dict) and (set(["decision", "instruction"])<=set(list(response.keys()))):
                    response = ReflectionOutput(decision=response["decision"], instruction=response["instruction"])
                    break
                else:
                    response = None
                    print(f"Retrying Iteration: {retry_ctr}")
                    continue
        assert response
        
        normalized = response.decision
        if (tool_name.lower() in ("user_input")) and (normalized.lower() in ("done")):
            print("ENFORCING `CONTINUE` after `user_input`")
            response.decision = "CONTINUE"
            normalized = response.decision
        
        reflection_id = len(state["messages"])
        agent_state = {
            "messages": [
                HumanMessage(content=f"[REFLECTION] {normalized}")
            ],
            "reflections": [
                AIMessage(content=normalized, additional_kwargs=response.model_dump())
            ],
            "reflection_ids": state.get("reflection_ids", [])+[reflection_id],
        }
        self.reflection_state = {
            "messages": state["messages"].append(
                HumanMessage(content=f"[REFLECTION] {normalized}")
            ),
            "reflections": state["reflections"].append(
                AIMessage(content=normalized, additional_kwargs=response.model_dump())
            ),
            "reflection_ids": state.get("reflection_ids", []).append([reflection_id]),
        }
        
        print("*"*60 + "\n")
        print(response)
        print(agent_state)
        print("*"*60 + "\n")
        return agent_state
    
    def should_continue(self, state: AgentState) -> Literal["tool_calling_node", "user_input_node", "__end__"]:
        """
        Routes after reflection_node.
        Uses STRICT prefix matching for deterministic routing.
        
        Checks reflection message:
        - If starts with "[REFLECTION] DONE" -> end
        - If starts with "[REFLECTION] CONTINUE:" -> loop back to tool_calling_node
        """
        print("\n" + "="*30)
        print("ROUTING: should_continue_after_reflection")
        print("="*30)
        
        messages = state["messages"]
        reflection_messages = state["reflections"]
        last_message = messages[-1]
        last_reflection_message = reflection_messages[-1]
        
        print(f" Total messages: {len(messages)}")
        print(last_message)
        print(f" Total reflections: {len(reflection_messages)}")
        print(last_reflection_message)
        
        if hasattr(last_reflection_message, 'additional_kwargs'):
            # Strict prefix matching with upper() for case-insensitive comparison
            decision = last_reflection_message.additional_kwargs.get("decision", None)
            instruction = last_reflection_message.additional_kwargs.get("instruction", None)
            
            try:
                if decision.lower()=="done":
                    print(" ROUTING TO: __end__")
                    print("="*30 + "\n")
                    return "__end__"
                elif decision.lower()=="continue":
                    print(" ROUTING TO: tool_calling_node")
                    print("="*30 + "\n")
                    return "tool_calling_node"
                elif decision.lower()=="user_input":
                    print(" ROUTING TO: user_input")
                    print("="*30 + "\n")
                    return "user_input_node"
                else:
                    raise ValueError("No decision reached by the reflection agent.")
            except Exception as e:
                print(f"Reflection failed with error: {e}")
                return "__end__"