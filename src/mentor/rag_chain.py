import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss

from src.config import (
    CAREER_NOTES_DIR,
    JOBS_CSV_PATH,
    CAREER_FAISS_INDEX,
    CAREER_METADATA_PKL,
    DEFAULT_RAG_TOP_K,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    DEFAULT_LLM_PROVIDER,
    LLM_MODEL_NAME
)
from src.parsing.loader import DocumentLoader
from src.search.embed import EmbeddingManager
from src.generate.prompts import MENTOR_RAG_SYSTEM_PROMPT

class CareerDocumentChunk:
    def __init__(self, doc_name: str, chunk_id: int, content: str, doc_type: str = "career_note"):
        self.doc_name = doc_name
        self.chunk_id = chunk_id
        self.content = content
        self.doc_type = doc_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_name": self.doc_name,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "doc_type": self.doc_type
        }

class CareerMentorRAG:
    """RAG-based AI Career Mentor grounded in career notes and job corpus."""

    def __init__(
        self,
        index_path: Path = CAREER_FAISS_INDEX,
        metadata_path: Path = CAREER_METADATA_PKL,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedder = EmbeddingManager.get_instance()
        self.provider = (provider or DEFAULT_LLM_PROVIDER).lower()
        self.model = model or LLM_MODEL_NAME
        self.index: Optional[faiss.Index] = None
        self.chunks: List[CareerDocumentChunk] = []

    def index_exists(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists()

    def build_knowledge_base(self, force_rebuild: bool = False) -> int:
        """
        Load career notes and job descriptions, chunk them, embed, and store in FAISS.
        """
        if self.index_exists() and not force_rebuild:
            self.load_knowledge_base()
            return len(self.chunks)

        all_chunks: List[CareerDocumentChunk] = []

        # 1. Ingest Career Notes (.md, .txt)
        if CAREER_NOTES_DIR.exists():
            for doc_file in sorted(CAREER_NOTES_DIR.glob("*.*")):
                if doc_file.suffix.lower() in [".md", ".txt"]:
                    try:
                        content = DocumentLoader.load_text(doc_file)
                        text_chunks = DocumentLoader.chunk_text(content, chunk_size=500, chunk_overlap=80)
                        for i, ch in enumerate(text_chunks):
                            all_chunks.append(CareerDocumentChunk(
                                doc_name=doc_file.name,
                                chunk_id=i,
                                content=ch,
                                doc_type="career_note"
                            ))
                    except Exception as e:
                        print(f"[CareerMentorRAG] Error loading {doc_file.name}: {e}")

        # 2. Ingest sample job summaries into knowledge base
        if JOBS_CSV_PATH.exists():
            try:
                import pandas as pd
                df = pd.read_csv(JOBS_CSV_PATH)
                for _, row in df.iterrows():
                    summary = f"Job Title: {row.get('title')}. Required Skills: {row.get('skills')}. Overview: {row.get('description')}"
                    all_chunks.append(CareerDocumentChunk(
                        doc_name="Industry Job Market Corpus",
                        chunk_id=int(row.get('job_id', 0)),
                        content=summary,
                        doc_type="job_posting"
                    ))
            except Exception as e:
                print(f"[CareerMentorRAG] Error loading jobs corpus: {e}")

        if not all_chunks:
            # Fallback chunk
            all_chunks.append(CareerDocumentChunk(
                doc_name="Default Guidance",
                chunk_id=0,
                content="Focus on building end-to-end projects, mastering Python, SQL, and data structures.",
                doc_type="general"
            ))

        # Embed all chunks
        texts = [c.content for c in all_chunks]
        embeddings = self.embedder.embed_documents(texts)
        dim = embeddings.shape[1]

        # FAISS Cosine Index
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))

        with open(self.metadata_path, "wb") as f:
            pickle.dump(all_chunks, f)

        self.index = index
        self.chunks = all_chunks
        return len(all_chunks)

    def load_knowledge_base(self):
        """Load FAISS index and chunk metadata."""
        if not self.index_exists():
            self.build_knowledge_base()
            return

        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.chunks = pickle.load(f)

    def retrieve(self, query: str, top_k: int = DEFAULT_RAG_TOP_K) -> List[Tuple[CareerDocumentChunk, float]]:
        """Retrieve most relevant knowledge base chunks with similarity scores."""
        if self.index is None or not self.chunks:
            self.load_knowledge_base()

        query_vec = self.embedder.embed_text(query)
        query_vec = np.expand_dims(query_vec, axis=0)

        scores, indices = self.index.search(query_vec, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results

    def answer(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = DEFAULT_RAG_TOP_K
    ) -> Dict[str, Any]:
        """
        End-to-end RAG answer generation:
        1. Retrieve top context
        2. Format prompt
        3. Invoke LLM or structured fallback
        4. Return answer + sources
        """
        retrieved = self.retrieve(query, top_k=top_k)

        # Build context string
        context_blocks = []
        sources = []
        for chunk, score in retrieved:
            context_blocks.append(f"[{chunk.doc_name} | Relevancy: {score:.2f}]\n{chunk.content}")
            sources.append({
                "doc_name": chunk.doc_name,
                "score": round(score, 3),
                "snippet": chunk.content[:160] + "..."
            })

        context_str = "\n\n---\n\n".join(context_blocks)
        system_prompt = MENTOR_RAG_SYSTEM_PROMPT.format(context=context_str)

        # 1. Try Gemini
        if self.provider == "gemini" and GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                # Incorporate chat history
                messages = []
                if chat_history:
                    for m in chat_history[-4:]:
                        role = "user" if m.get("role") == "user" else "model"
                        messages.append({"role": role, "parts": [m.get("content", "")]})
                
                prompt_full = f"{system_prompt}\n\nUSER QUESTION: {query}"
                response = model.generate_content(prompt_full)
                return {
                    "answer": response.text.strip(),
                    "sources": sources,
                    "retrieved_count": len(retrieved),
                    "provider": "gemini"
                }
            except Exception as e:
                print(f"[CareerMentorRAG] Gemini generation failed: {e}. Falling back.")

        # 2. Try OpenAI
        if (self.provider == "openai" or OPENAI_API_KEY) and OPENAI_API_KEY:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                messages = [{"role": "system", "content": system_prompt}]
                if chat_history:
                    for m in chat_history[-4:]:
                        role = "user" if m.get("role") == "user" else "assistant"
                        messages.append({"role": role, "content": m.get("content", "")})
                messages.append({"role": "user", "content": query})

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.3
                )
                return {
                    "answer": response.choices[0].message.content.strip(),
                    "sources": sources,
                    "retrieved_count": len(retrieved),
                    "provider": "openai"
                }
            except Exception as e:
                print(f"[CareerMentorRAG] OpenAI generation failed: {e}. Falling back.")

        # 3. Grounded Fallback Synthesizer (Ensures reliable answers without API keys)
        return self._grounded_synthesis_fallback(query, retrieved, sources)

    def _grounded_synthesis_fallback(
        self,
        query: str,
        retrieved: List[Tuple[CareerDocumentChunk, float]],
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synthesize a structured, grounded answer from retrieved chunks."""
        top_chunk = retrieved[0][0] if retrieved else None
        
        answer_parts = [
            f"### Career Mentor Guidance",
            f"Here is targeted guidance grounded in our career knowledge base and market requirements:\n"
        ]

        if top_chunk:
            answer_parts.append(f"**Key Insights from `{top_chunk.doc_name}`:**\n")
            # Extract key lines from the top chunk
            lines = [l.strip() for l in top_chunk.content.split("\n") if l.strip()]
            for line in lines[:6]:
                if line.startswith("#"):
                    answer_parts.append(f"**{line.replace('#', '').strip()}**")
                elif line.startswith("-") or line.startswith("*"):
                    answer_parts.append(line)
                else:
                    answer_parts.append(f"- {line}")

        answer_parts.append("\n**Recommended Next Steps:**")
        answer_parts.append("1. **Skill Gap Assessment:** Review the skills listed above against your current CV.")
        answer_parts.append("2. **Hands-On Capstone:** Build and deploy a public GitHub project showcasing these competencies.")
        answer_parts.append("3. **Resume Alignment:** Re-write your experience bullets using the STAR / Google X-Y-Z formula.")

        return {
            "answer": "\n".join(answer_parts),
            "sources": sources,
            "retrieved_count": len(retrieved),
            "provider": "grounded_local_retrieval"
        }
