import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

open_api_key_value = os.getenv('OPENAI_API_KEY')
if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    vectorstore = FAISS.from_texts([
        "How many Films were there in 2015 where Sumit was an actor",
        "How many Films were there during Covid pandemic where Sumit was an actor",
        "How many Film did Sumit released on 2010",
        "Who has acted in most hindi films in 2000",
        "List all films that Abhi released during Covid Pndemic",
        "What are the best films of",
        "Who has acted in most hindi films in 2000",
        "How many hindi movies got released in 2020?",
        "How many hindi movies got released in 2021?",
        "How many Bengali movies got released in 2020",
        "Which Country did Abhi live in 2021"],embeddings,distance_strategy=DistanceStrategy.COSINE)
    vectorstore.save_local("faiss_index_SQL")
    new_vectorstore = FAISS.load_local("faiss_index_SQL", embeddings,allow_dangerous_deserialization =True)
    response = embeddings.embed_query("How many films got released  between 2019 and 2021 for Sumit")
    result  = new_vectorstore.similarity_search_with_score_by_vector(response, 4)
    print(result)