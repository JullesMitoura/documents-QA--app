from pydantic import BaseModel
from typing import Optional

class QuestionRequest(BaseModel):
    question: str
    index_name: str
    top_k: int = 5

class CreateIndexRequest(BaseModel):
    index_name: str
    vector_dimension: int = 1536

class DeleteIndexRequest(BaseModel):
    index_name: str