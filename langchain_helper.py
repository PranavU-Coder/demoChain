from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

def generate_pet_name(animal_type,pet_color):
    
    llm = ChatOllama(
        model="llama3.2",  
        temperature=0.5
    )
    
    prompt_template = PromptTemplate(
        input_variables=['animal_type'],
        template="I have a {animal_type} pet and I want a cool name for it, it is {pet_color} in color. Suggest some cool names for it."
    )
    
    chain = prompt_template | llm | StrOutputParser()
    
    response = chain.invoke({'animal_type': animal_type, 'pet_color' : pet_color})
    return response

def langchain_agent():

    llm = ChatOllama(
        model="llama3.2",  
        temperature=0.5
    )
    
    tools = load_tools(["wikipedia","llm-math"],llm=llm)
    
    agent = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    result = agent.run(
        "what is the average age of a human being? Could You Multiply it with 6"
    )

    print(result)

if __name__ == '__main__':
    langchain_agent()