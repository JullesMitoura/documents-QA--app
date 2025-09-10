# function to search similar documents using AzureAI Search
from src.services import AzureSearchService
from src.services import OpenAIService
from src.utils.prompts import Prompts

def similar_search(azure_search_service: AzureSearchService,
                   query: str, 
                   index_name: str, 
                   top_k: int = 10):
    
    results = azure_search_service.get_similar(index_name=index_name, 
                                               query=query, 
                                               top_k=top_k)
    extracted_texts = [chunk["textual_content"] for chunk in results]
    full_text = " ".join(extracted_texts)
    return full_text

def get_response(openai_service: OpenAIService,
                 azure_search_service: AzureSearchService,
                 query: str,
                 index_name: str,
                 top_k: int = 10) -> str:
    similar_docs = similar_search(azure_search_service, 
                                  query, 
                                  index_name, 
                                  top_k)
    
    # generate response based on similar context
    prompt = Prompts.final_response(similar_docs, query)
    response = openai_service.invoke(prompt)
    return response