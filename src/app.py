from functools import lru_cache
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import Optional
from src.models.models import QuestionRequest, CreateIndexRequest, DeleteIndexRequest
from src.functions import get_response, create_index, upload_documents, delete_index
from src.services import AzureSearchService, OpenAIService
from src.utils import Settings
import shutil
import os

app = FastAPI(title="Document Q&A API")

# Cache services
@lru_cache(maxsize=1)
def get_services():
    sets = Settings()
    openai_service = OpenAIService(sets=sets)
    azure_search_service = AzureSearchService(embedding_model=openai_service, sets=sets)
    return openai_service, azure_search_service

openai_service, azure_search_service = get_services()

# API endpoint
@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        response = get_response(
            openai_service=openai_service,
            azure_search_service=azure_search_service,
            query=request.question,
            index_name=request.index_name,
            top_k=request.top_k
        )
        return {"question": request.question, "answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/create-index")
def api_create_index(request: CreateIndexRequest):
    try:
        create_index(
            index_name=request.index_name,
            vector_dimension=request.vector_dimension,
            azure_search_service=azure_search_service
        )
        return {"status": "success", "index_name": request.index_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-index")
def api_delete_index(request: DeleteIndexRequest):
    try:
        delete_index(
            index_name=request.index_name,
            azure_search_service=azure_search_service
        )
        return {"status": "success", "index_name": request.index_name, "message": "Index deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-document")
def api_upload_document(
    file: UploadFile = File(...),
    index_name: str = Form(...),
    processing_mode: str = Form("normal"),
    additional_information: Optional[str] = Form(None),
    library_name: Optional[str] = Form("default")
):
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Upload document
        upload_documents(
            index_name=index_name,
            document=temp_path,
            openai_service=openai_service,
            azure_search_service=azure_search_service,
            processing_mode=processing_mode,
            additional_information=additional_information,
            library_name=library_name
        )

        # Remove temporary file
        os.remove(temp_path)

        return {"status": "success", "file_name": file.filename, "index_name": index_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))