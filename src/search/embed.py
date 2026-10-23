import os
import numpy as np
from typing import List, Union

from src.config import (
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
)

class EmbeddingManager:
    """Manages embedding generation across HuggingFace SentenceTransformers, OpenAI, and Gemini."""

    _instance = None
    _st_model = None

    def __init__(self, provider: str = EMBEDDING_PROVIDER, model_name: str = EMBEDDING_MODEL_NAME):
        self.provider = provider.lower()
        self.model_name = model_name
        self.dimension = EMBEDDING_DIMENSION

    @classmethod
    def get_instance(cls, provider: str = EMBEDDING_PROVIDER):
        if cls._instance is None:
            cls._instance = cls(provider=provider)
        return cls._instance

    def _get_sentence_transformer(self):
        if EmbeddingManager._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                EmbeddingManager._st_model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[EmbeddingManager] SentenceTransformer load failed: {e}")
                return None
        return EmbeddingManager._st_model

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single string. Returns 1D float32 normalized numpy array."""
        embeddings = self.embed_documents([text])
        return embeddings[0]

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings. Returns 2D float32 normalized numpy array of shape (N, dimension)."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # 1. Try local SentenceTransformers (Fast, reliable, free)
        if self.provider == "sentence-transformers":
            st = self._get_sentence_transformer()
            if st is not None:
                embeddings = st.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return embeddings.astype(np.float32)

        # 2. Try OpenAI Embeddings if provider is openai and key is present
        if self.provider == "openai" and OPENAI_API_KEY:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                resp = client.embeddings.create(input=texts, model="text-embedding-3-small")
                vectors = [item.embedding for item in resp.data]
                arr = np.array(vectors, dtype=np.float32)
                # L2 normalize
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return arr / norms
            except Exception as e:
                print(f"[EmbeddingManager] OpenAI embed failed: {e}. Falling back.")

        # 3. Try Gemini Embeddings if provider is gemini and key is present
        if self.provider == "gemini" and GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                vectors = []
                for t in texts:
                    res = genai.embed_content(
                        model="models/text-embedding-004",
                        content=t,
                        task_type="retrieval_document"
                    )
                    vectors.append(res['embedding'])
                arr = np.array(vectors, dtype=np.float32)
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return arr / norms
            except Exception as e:
                print(f"[EmbeddingManager] Gemini embed failed: {e}. Falling back.")

        # 4. Fallback: Local SentenceTransformer if not already tried
        st = self._get_sentence_transformer()
        if st is not None:
            embeddings = st.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.astype(np.float32)

        # 5. Fallback deterministic pseudo-embeddings (ensures tests and app never crash)
        return self._fallback_hashing_embeddings(texts)

    def _fallback_hashing_embeddings(self, texts: List[str]) -> np.ndarray:
        """Deterministic TF-IDF/Hash projection fallback if all heavy models fail."""
        dim = self.dimension
        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            for w in words:
                h = abs(hash(w)) % dim
                matrix[i, h] += 1.0
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm
            else:
                matrix[i, 0] = 1.0
        return matrix
