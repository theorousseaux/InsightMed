from langchain_core import embeddings
import requests
from typing import Dict, List
from ..config import base_url, openwebui_api_key, embedding_model


class My_embeddings(embeddings.Embeddings):
    def __init__(self, model: str = "mxbai-embed-large:latest"):
        self.model = model

    def get_embeddings_from_text(self, text: str) -> Dict[str, List[float]]:
        response = requests.post(
            url=base_url + "/api/embeddings",
            headers={"Authorization": f"Bearer {openwebui_api_key}"},
            json={"model": self.model, "prompt": text},
        )
        return response.json()

    def embed_query(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        return self.get_embeddings_from_text(text)["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.get_embeddings_from_text(text)["embedding"])
        return embeddings


my_embeddings = My_embeddings(model=embedding_model)
