import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai import OpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.globals import set_debug


open_api_key_value = os.getenv('OPENAI_API_KEY')
#setting debug logs to check the request flow to LLM
set_debug(True)
os.environ["LANGCHAIN_TRACING_MODE"]="console"

llm = OpenAI(
    openai_api_key = open_api_key_value,
    temperature=0,
    model="gpt-3.5-turbo-instruct"
)

if __name__ == "__main__":
    score_tolerance = 0.003
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    response = embeddings.embed_query("How many films got released  between 2019 and 2021 where Sumit was an actor")

    vectorstore = FAISS.from_embeddings([("How many films got released  between 2019 and 2021 where Sumit was an actor",response[:5])],embeddings,distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    response1 = embeddings.embed_query("How many films got released  between 2019 and 2021 where Abhishek was an actor")
    vectorstore.add_embeddings([("How many films got released  between 2019 and 2021 where Abhishek was an actor",response1[:5])])
    response3 = embeddings.embed_query("5 best films where Debnath was a director")
   
    vectorstore.add_embeddings([("5 best films where Debnath was a director",response3[:5])])
    print(" All embedding loaded in the vectorstore")
   
    vectorstore.save_local("faiss_index_SQL")
    new_vectorstore = FAISS.load_local("faiss_index_SQL", embeddings,allow_dangerous_deserialization =True)
    response4 = embeddings.embed_query("How many films got released between 2019 and 2021 where Raj was actor")
    result  = new_vectorstore.similarity_search_with_score_by_vector(response4[:5], 4)
    print(result)
    #vectorstore = FAISS
    #vectorstore.save_local("faiss_index_react")
    #filgter results based on score tolerance
    filtered_results = [item for item in result if item[1] >= score_tolerance]
    # print("Filtered results based on score tolerance:", filtered_results)
    for result in filtered_results:
        document, score = result  # Unpacking the tuple
        page_content = document.page_content
        
    
    print("The page content is:",page_content)
    input_data = {"How many films got released between 2019 and 2021 where Subhamoy was an actor"}


    template  = """

    Example1: 
USER_QUESTION: How many films got released  between 2019 and 2021 where Sumit was an actor
answer as SQL_RESULT: Select count (*) from MY_TX_FILMS where release_year BETWEEN '2019' AND '2021' and actor='Sumit '
Example2:
USER_QUESTION: How many films got released  between 2019 and 2021 where Abhishek was an actor
answer as SQL_RESULT: Select count (*) from MY_TX_FILMS where release_year BETWEEN '2019' AND '2021' and actor='abhisek '
USER_QUESTION: ${input_data}
"""
prompt = PromptTemplate.from_template(template)
print('The prompt is ',prompt)
llm_chain = LLMChain(prompt=prompt, llm=llm)

# input_data = {"USER_QUESTION": "How many films got released between 2019 and 2021 where Raj was an actor"}

langchain_response = llm_chain.invoke(input=input_data)
print("the response is",langchain_response)

