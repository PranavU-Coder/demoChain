# imports

from typing import *

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END


# the class we are defining for the chatbot attributes we will be requiring

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatOllama(model="llama3.2")

# the node in the graph

def process_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"content: {response.content}")
    return state 

# constructing the graph

graph = StateGraph(AgentState)

graph.add_node("processor", process_node)

graph.add_edge(START, "processor")
graph.add_edge("processor", END)

agent = graph.compile()

# results

user_input = input("enter: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("enter: ")