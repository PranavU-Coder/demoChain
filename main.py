from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_pet_name(animal_type):
    
    llm = ChatOllama(
        model="llama3.2",  
        temperature=0.5
    )
    
    prompt_template = PromptTemplate(
        input_variables=['animal_type'],
        template="I have a {animal_type} pet and I want a cool name for it. Suggest some cool names for it."
    )
    
    chain = prompt_template | llm | StrOutputParser()
    
    response = chain.invoke({'animal_type': animal_type})
    return response

print(generate_pet_name('cat'))
