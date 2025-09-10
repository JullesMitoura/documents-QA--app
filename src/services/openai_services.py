from openai import AzureOpenAI, AsyncAzureOpenAI
from typing import Any, Dict, List, Union
from src.utils import Settings
from openai.types.chat.chat_completion_message import ChatCompletionMessage


class OpenAIService:
    """
    Service wrapper for Azure OpenAI (sync and async).
    
    Main methods:
    - invoke: synchronous chat model call.
    - async_invoke: asynchronous chat model call.
    - embed: synchronous embeddings generation.
    - async_embed: asynchronous embeddings generation.
    """

    def __init__(self, 
                 sets: Settings,
                 timeout: int = 60,
                 max_retries: int = 2):
        """
        Initialize the Azure OpenAI clients (sync and async).
        """
        self.common_args = {
            "api_key": sets.azure_openai_api_key,
            "azure_endpoint": sets.azure_openai_endpoint,
            "api_version": sets.llm_api_version,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        self.llm_deployment = sets.llm_deployment_model
        self.embedding_deployment = sets.embedding_deployment_model
        self.sync_client = AzureOpenAI(**self.common_args)
        self.async_client = AsyncAzureOpenAI(**self.common_args)

    def _prepare_messages(self, prompt: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Convert a string or list of messages into the API-compatible format.
        """
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, list):
            for msg in prompt:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    raise ValueError("Each message must be a dict with 'role' and 'content'.")
            return prompt
        raise ValueError("The prompt must be a string or a list of message dictionaries.")

    def invoke(self, 
               prompt: Union[str, List[Dict[str, Any]]], 
               **kwargs) -> ChatCompletionMessage:
        """
        Synchronous call to the chat model.
        """
        messages = self._prepare_messages(prompt)
        response = self.sync_client.chat.completions.create(
            model=self.llm_deployment,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

    async def ainvoke(self, 
                           prompt: Union[str, List[Dict[str, Any]]], 
                           **kwargs) -> ChatCompletionMessage:
        """
        Asynchronous call to the chat model.
        """
        messages = self._prepare_messages(prompt)
        response = await self.async_client.chat.completions.create(
            model=self.llm_deployment,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

    def embed(self, prompt: str) -> List[List[float]]:
        """
        Generate embeddings synchronously.
        Returns a list of embeddings (one per input).
        """
        response = self.sync_client.embeddings.create(
            model=self.embedding_deployment,
            input=prompt
        )
        return [item.embedding for item in response.data]

    async def aembed(self, prompt: str) -> List[List[float]]:
        """
        Generate embeddings asynchronously.
        Returns a list of embeddings (one per input).
        """
        response = await self.async_client.embeddings.create(
            model=self.embedding_deployment,
            input=prompt
        )
        return [item.embedding for item in response.data]