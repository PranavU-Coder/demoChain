from typing import *

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage

from operator import add as add_messages

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_core.tools import tool 

llm = ChatOllama(model="llama3.2", temperature = 0) 
embeddings = OllamaEmbeddings(model="nomic-embed-text")

pdf = "market.pdf"
pdf_loader = PyPDFLoader(pdf)

try:
    pages = pdf_loader.load()
    print(f"pdf has been loaded and is of {len(pages)} pages.")
except Exception as e:
    print(f"error loading pdf: {e}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

pages_split = text_splitter.split_documents(pages)

persist_dir = r'/home/pranav/Projects/demoChain/lang_graph/ai_agents' 
collection_name = "stock_market"

# creating a chroma vector database

try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name
    )
    print('created a vector database')
except Exception as e:
    print(f'error setting up chromadb: {str(e)}')
    raise

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

@tool
def retriever_tool(query: str) -> str:
    """
    tool searches and returns the information from stock market document from year 2024.
    """ 

    docs = retriever.invoke(query)

    if not docs:
        return "i have not found any relevant information regarding this topic in the document."
    results = []
    for i, doc in enumerate(docs):
        results.append(f'document {i+1}:\n{doc.page_content}')

    return "\n\n".join(results)

tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """
    check if the last message contains tool calls.
    """

    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {tool.name: tool for tool in tools}

def call_llm(state: AgentState) -> AgentState:
    """
    function to call the LLM with the current state.
    """

    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}

def take_action(state: AgentState) -> AgentState:
    """
    execute tool calls from the LLM responses.
    """

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f'calling tool: {t['name']} with query: {t['args'].get('query', 'no query provided')}')

        if not t['name'] in tools_dict:
            print(f'\ntool: {t['name']} doesnt exist.')
            result = "incorrect tool name, please retry and select tool from a list of available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f'result length: {len(str(result))}')

        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

        print('tool execution complete, going back to model')
        return {'messages': results}

graph = StateGraph(AgentState)

graph.add_node('llm', call_llm)
graph.add_node('retriever', take_action)

graph.add_edge(START, 'llm')

graph.add_conditional_edges(
    'llm',
    should_continue,
    {
        True: 'retriever',
        False: END
    }
)

graph.add_edge('retriever', 'llm')

agent = graph.compile()

def running_agent():
    print('\n == RAG AGENT ==')

    while True:
        user_input = input('q: ')
        if user_input.lower() in ['exit', 'quit']:
            break
        messages = [HumanMessage(user_input)]
        result = agent.invoke({'messages': messages})

        print('\n == ANSWER ==')
        print(result['messages'][-1].content)

if __name__ == '__main__':
    running_agent()
