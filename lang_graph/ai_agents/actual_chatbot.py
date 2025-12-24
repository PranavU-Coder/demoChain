import os
from typing import *

from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOllama(model="llama3.2")

def process_node(state: AgentState) -> AgentState:
    """
    solve the request that has been inputted.
    """
    response = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    print(f"AI: {response.content}")

    return state 

graph = StateGraph(AgentState)

graph.add_node("processor", process_node)

graph.add_edge(START, "processor")
graph.add_edge("processor", END)

agent = graph.compile()

conversation_history = []

user_input = input("enter: ")

while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})

    conversation_history = result["messages"]

    user_input = input("enter: ")

with open("logging.txt", "w") as file:
    file.write("conversation logs:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"person: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("end of conversation\n")
