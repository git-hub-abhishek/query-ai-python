import os
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.globals import set_debug
import json
from langchain_openai import OpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

open_api_key_value = os.getenv('OPENAI_API_KEY')
llm = OpenAI(
    openai_api_key = open_api_key_value,
    temperature=0,
    model="gpt-3.5-turbo-instruct"
)

#setting debug logs to check the request flow to LLM
# set_debug(True)
os.environ["LANGCHAIN_TRACING_MODE"]="console"
embedding_length =512
embeddings = OpenAIEmbeddings(model="text-embedding-3-large") 

#Truncating the response embedding to a fixed embedding length
def truncate_embedding(response,embedding_length):
    if embedding_length>0:
        truncated_response = response[:embedding_length]
    else:
        truncated_response = response
    return truncated_response

#Method to create response embeddings for the description value of the table
def create_response_embeddings(description):
    response  = embeddings.embed_query(description)
    return response

if __name__ == "__main__":
    # Load the spreadsheet file
    file_path = 'D:\\Imp-LLM-file.xlsx'

   # Read the Excel file
    df = pd.read_excel(file_path)

    #Creating a dictionary that contains value description and table_name from the excel-file
    description_table_dict = pd.Series(df['Description'].values,index=df['Table_Name']).to_dict()

    # Dictionary to store the responses with the table name as key
    table_embeddings_dict = {}

    for table_name, description in description_table_dict.items():
        response = create_response_embeddings(description)

        truncated_response = truncate_embedding(response, embedding_length)
        table_embeddings_dict[table_name] = truncated_response
    
    first_table_name, first_embedding = next(iter(table_embeddings_dict.items()))
    print(first_table_name)
    vectorstore = FAISS.from_embeddings([(first_table_name, first_embedding)], embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    
    for i, (table, embedding) in enumerate(table_embeddings_dict.items()):
    # The embeddings are already truncated when added to table_embeddings_dict
    # Assuming that the embeddings are lists or arrays that can be sliced
         if i > 0:             
             print(table)
             vectorstore.add_embeddings([(table, embedding)])
             
    
    prompt = "Which film did the customers from country Africa rented the most"
    #"what are tables needed to find out films of which language got rented maximum number of times in 2022"
    #"Which film did the customers from England rented the most"
    print("The prompt is:",prompt)
    response_prompt = embeddings.embed_query(prompt)
    
    closest_table_names_and_score  = vectorstore.similarity_search_with_score_by_vector(response_prompt[:512],20)
    print("Prinitng the table name",closest_table_names_and_score)


    closest_table_names = []

    for result in closest_table_names_and_score:
        document, score = result  # Unpacking the tuple
        page_content = document.page_content
        closest_table_names.append(page_content)
        
    print(closest_table_names)
    #The closest_table_names consists of trailing white spaces removing the training whitespaces
    cleaned_closest_table_names = [name.strip() for name in closest_table_names]


    # Step 1: Read the table_metadata.json file
    with open('table_metadata.json', 'r') as file:
        table_metadata = json.load(file)   
    
    
    filtered_metadata = {key: value for key, value in table_metadata.items() if key.strip() in cleaned_closest_table_names}
    json_string =""
    # with open('filtered_tables.json', 'w') as file:
        
    json_string  = json.dumps(filtered_metadata, indent=4)
    
    template = """You are an expert DBA, write a SINGLE SQL query to do {userQuestion}.
      Your answer must consider the db_lookup_schema:{dbSqlSchema}.
      Instruction: You must respond with {{"SQL": "I DONT KNOW"}} if you are not sure about the answer.
      Instruction: You must consider the Enum defined in the db_lookup_schema.
      Instruction: Your answer MUST MUST MUST have at least one table defined in the db_lookup_schema. else respond with {{"SQL": "I DONT KNOW"}}."""
    
    dbSqlSchema = json_string
    userQuestion = prompt

    prompt_template = PromptTemplate(
        input_variables=["userQuestion", "dbSqlSchema"], 
        template=template
    )
    

    llm_chain = LLMChain(prompt=prompt_template, llm=llm)    
   
    # Construct the input_dict here
    input_dict = {
        "userQuestion": prompt,
        "dbSqlSchema": json_string
    }   
    langchain_response = llm_chain.invoke(input=input_dict)
    print("#####################################################################################################")
    print("Printing the langchain response:",langchain_response['text'])