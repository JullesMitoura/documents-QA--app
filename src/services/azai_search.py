from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)
from src.utils import Settings
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class AzureSearchService:
    def __init__(self, 
                 embedding_model, 
                 sets: Settings):
        self.embedding_model = embedding_model
        self.azai_url = sets.azure_ai_search_endpoint
        self.azai_key = sets.azure_ai_search_key
        self.credential = AzureKeyCredential(self.azai_key)

    def delete_index(self, index_name: str):
        logger.info(f"Deleting index '{index_name}'...")
        index_client = SearchIndexClient(endpoint=self.azai_url, credential=self.credential)
        try:
            index_client.delete_index(index_name)
            logger.info(f"Index '{index_name}' deleted successfully.")
        except HttpResponseError as e:
            if e.status_code == 404:
                logger.warning(f"Index '{index_name}' not found.")
            else:
                logger.error(f"Error deleting index '{index_name}': {str(e)}")
                raise
        
    def create_index(self, 
                     index_name: str, 
                     embedding_dimensions: int = 1536, 
                     recreate_if_exists: bool = False):
        logger.info(f"Creating index '{index_name}'...")
        
        algorithm_config_name = "myHnswConfig"
        profile_name = "myHnswProfile"
        
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name="textual_content", type=SearchFieldDataType.String, searchable=True),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=embedding_dimensions,
                vector_search_profile_name=profile_name,
            ),
            SimpleField(name="library", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="created_date", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SearchableField(name="title", type=SearchFieldDataType.String, searchable=True),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        ]
        
        vector_search = VectorSearch(
            profiles=[VectorSearchProfile(
                name=profile_name, 
                algorithm_configuration_name=algorithm_config_name
            )],
            algorithms=[HnswAlgorithmConfiguration(
                name=algorithm_config_name,
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine" 
                }
            )]
        )
        
        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
        index_client = SearchIndexClient(endpoint=self.azai_url, credential=self.credential)
        
        index_exists = index_name in index_client.list_index_names()
        
        if index_exists:
            if recreate_if_exists:
                logger.info(f"Deleting existing index '{index_name}'...")
                index_client.delete_index(index_name)
                logger.info(f"Creating new index '{index_name}'...")
                result = index_client.create_index(index)
            else:
                try:
                    logger.info(f"Attempting to update existing index '{index_name}'...")
                    result = index_client.create_or_update_index(index)
                except HttpResponseError as e:
                    if "Algorithm name cannot be updated" in str(e):
                        logger.error("Cannot update vector algorithm configuration. Please delete the index first or use a new name.")
                        raise ValueError("Cannot update vector algorithm configuration. Set recreate_if_exists=True or use a new index name.") from e
                    raise
        else:
            result = index_client.create_index(index)
        
        logger.info(f"Index '{result.name}' operation completed successfully")
        return result

    def upload_documents(self, index_name: str, documents: list, batch_size: int = 100):
        logger.info(f"Uploading documents to index '{index_name}'...")
        
        search_client = SearchClient(endpoint=self.azai_url, 
                                   index_name=index_name, 
                                   credential=self.credential)
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                result = search_client.upload_documents(documents=batch)
                logger.info(f"Uploaded batch {i//batch_size + 1}: {len(result)} documents")
            except Exception as e:
                logger.error(f"Error uploading batch {i//batch_size + 1}: {str(e)}")
                raise
        
        logger.info(f"Successfully uploaded {len(documents)} documents")

    def get_similar(self, index_name: str, query: str, top_k: int = 5, filter: str = None):
        logger.info(f"Searching in index '{index_name}' for: {query}")
        
        search_client = SearchClient(endpoint=self.azai_url, 
                                index_name=index_name, 
                                credential=self.credential)
        
        vector = self.embedding_model.embed(query)[0]
        results = search_client.search(
            search_text=query,
            vector_queries=[
                {
                    "vector": vector,
                    "fields": "content_vector",
                    "k": top_k,
                    "kind": "vector",
                    "exhaustive": True
                }
            ],
            top=top_k,
            filter=filter,
            select=["id", 
                    "textual_content", 
                    "title", 
                    "library", 
                    "source", 
                    "created_date"]
        )
        
        return list(results)