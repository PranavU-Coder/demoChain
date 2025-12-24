from typing import *
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool 
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode 

class AgentState(TypedDict):
     messages: Annotated[Sequence[BaseMessage], add_messages]

@tool 
def add(a: int, b: int) -> int:
    """
    adds two integer values and outputs the resultant integer.
    """
    return a+b 

@tool 
def multiply(a: int, b: int) -> int:
    """
    multiplies two integer values and outputs the result integer.
    """
    return a*b

tools = [add]
model = ChatOllama(model="llama3.2").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    
    system_prompt = SystemMessage(content =
                                  "you're my assistant, and must comply to whatever i ask you to do.")

    response = model.invoke([system_prompt] + state['messages'])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state['messages']
    last_msg = messages[-1]
    if not last_msg.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)

graph.add_node("model", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "model")

graph.add_conditional_edges(
    "model",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "model")
agent = graph.compile()

input = {'messages': [("user", "add 6 and 9 together please and then multiply it with 10" )]}
print(agent.invoke(input))
