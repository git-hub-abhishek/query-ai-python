import os
import spacy
from spacy import displacy
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
open_api_key_value = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_MODE"]="console"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

nlp = spacy.load("en_core_web_trf")

def replace_actor_names(query):
    doc = nlp(query)
    modified_query = query
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            modified_query = modified_query.replace(ent.text, "[actor]")
    
    return modified_query

if __name__ == "__main__":
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    vectorstore = FAISS.from_texts(["How many films got released  between 2019 and 2021 for Sumit",
                                    "How many films got released  between 2019 and 2021 for Abhisek",
                                    "How many films got released  between 2019 and 2021 for [actor]"
        ],embeddings,distance_strategy=DistanceStrategy.COSINE)
    vectorstore.save_local("faiss_index_SQL")
    new_vectorstore = FAISS.load_local("faiss_index_SQL", embeddings,allow_dangerous_deserialization =True)
    
    query = "How many films got released  between 2019 and 2021 for Tom"
    modified_query = replace_actor_names(query)
    print('the modified query is:',modified_query)
    response = embeddings.embed_query(modified_query)
    result  = new_vectorstore.similarity_search_with_score_by_vector(response, 4)
    print(result)