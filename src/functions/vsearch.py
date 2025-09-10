# function to create a new index using AzureAI Search
from src.services import AzureSearchService
from src.services import OpenAIService
from src.utils.extractor import process_document
from src.utils.reader import process_document_images
from src.utils.chunker import chunker
from datetime import datetime
import uuid
import os

def create_index(index_name: str,
                 vector_dimension: int,
                 azure_search_service: AzureSearchService) -> None:
    
    # Create the index
    azure_search_service.create_index(index_name=index_name, embedding_dimensions=vector_dimension)

def delete_index(index_name: str,
                 azure_search_service: AzureSearchService) -> None:
    azure_search_service.delete_index(index_name=index_name)

def delete_index(index_name: str,
                 azure_search_service: AzureSearchService) -> None:
    azure_search_service.delete_index(index_name=index_name)

def upload_documents(index_name: str,
                     document: str,
                     openai_service: OpenAIService,
                     azure_search_service: AzureSearchService,
                     processing_mode: str = "normal",
                     additional_information: str = None,
                     library_name: str = None) -> None:

    file_name = os.path.basename(document)
    # Process and upload documents
    # 1. extract text
    processed_doc = process_document(file_path=document, processing_mode=processing_mode)
    
    if processing_mode == "normal":
        full_text = " ".join(item["content"] for item in processed_doc)

    else:
        full_text = " "  
        if additional_information:
            images_text = process_document_images(document_content = processed_doc,
                                                    service=openai_service, 
                                                    document_informations=additional_information)
        else:
            images_text = process_document_images(document_content = processed_doc,
                                                    service=openai_service, 
                                                    document_informations=file_name)
        full_text += " " + images_text

    # 2. chunk text
    chunks = chunker(text=full_text, chunk_size=2000, overlap=200)

    # 3. upload chunks
    documents = []
    for i, chunk in enumerate(chunks, start=1):
        doc = {
            "id": str(uuid.uuid4()),  # unique ID
            "textual_content": chunk,
            "content_vector": openai_service.embed(chunk)[0],
            "library": library_name,
            "created_date": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "title": f"i - document: {file_name}",
            "source": "document_chunks"
        }
        documents.append(doc)

        azure_search_service.upload_documents(index_name=index_name, 
                                              documents=documents, 
                                              batch_size=100)