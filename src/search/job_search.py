import os
import pickle
import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import (
    JOBS_CSV_PATH,
    JOBS_FAISS_INDEX,
    JOBS_METADATA_PKL,
    DEFAULT_TOP_K_JOBS
)
from src.search.embed import EmbeddingManager

class JobMatchResult:
    """Represents a matched job with similarity scores and skill gap analysis."""

    def __init__(
        self,
        job_id: Any,
        title: str,
        company: str,
        location: str,
        experience_level: str,
        skills: str,
        description: str,
        score: float,
        matching_skills: List[str],
        missing_skills: List[str]
    ):
        self.job_id = job_id
        self.title = title
        self.company = company
        self.location = location
        self.experience_level = experience_level
        self.skills = skills
        self.description = description
        self.score = float(score)
        self.match_percentage = f"{max(0.0, min(100.0, self.score * 100)):.1f}%"
        self.matching_skills = matching_skills
        self.missing_skills = missing_skills

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "experience_level": self.experience_level,
            "skills": self.skills,
            "description": self.description,
            "score": round(self.score, 4),
            "match_percentage": self.match_percentage,
            "matching_skills": self.matching_skills,
            "missing_skills": self.missing_skills
        }

class JobSearchEngine:
    """FAISS-powered semantic search engine for matching candidate profiles against job postings."""

    def __init__(self, index_path: Path = JOBS_FAISS_INDEX, metadata_path: Path = JOBS_METADATA_PKL):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedder = EmbeddingManager.get_instance()
        self.index: Optional[faiss.Index] = None
        self.jobs_df: Optional[pd.DataFrame] = None

    def index_exists(self) -> bool:
        """Check if FAISS index and metadata files exist."""
        return self.index_path.exists() and self.metadata_path.exists()

    def build_index(self, csv_path: Optional[Path] = None, force_rebuild: bool = False) -> int:
        """
        Embed all job postings and save the FAISS vector index.
        Returns the number of jobs indexed.
        """
        if self.index_exists() and not force_rebuild:
            self.load_index()
            return len(self.jobs_df)

        path = csv_path or JOBS_CSV_PATH
        if not path.exists():
            raise FileNotFoundError(f"Jobs dataset not found at {path}")

        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("Jobs dataset is empty.")

        # Build rich text for semantic embedding
        search_texts = []
        for _, row in df.iterrows():
            title = str(row.get('title', ''))
            company = str(row.get('company', ''))
            skills = str(row.get('skills', ''))
            desc = str(row.get('description', ''))
            text = f"Job Title: {title}. Company: {company}. Required Skills: {skills}. Description: {desc}"
            search_texts.append(text)

        # Generate L2-normalized embeddings
        embeddings = self.embedder.embed_documents(search_texts)
        dimension = embeddings.shape[1]

        # Use Inner Product (cosine similarity on normalized vectors)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        # Save to disk
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))

        with open(self.metadata_path, "wb") as f:
            pickle.dump(df, f)

        self.index = index
        self.jobs_df = df
        return len(df)

    def load_index(self):
        """Load FAISS index and metadata dataframe from disk."""
        if not self.index_exists():
            self.build_index()
            return

        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.jobs_df = pickle.load(f)

    def search_jobs(
        self,
        query_text: str,
        candidate_skills: Optional[List[str]] = None,
        top_k: int = DEFAULT_TOP_K_JOBS
    ) -> List[JobMatchResult]:
        """
        Perform semantic similarity search against indexed jobs.
        """
        if self.index is None or self.jobs_df is None:
            self.load_index()

        # Embed candidate query / profile
        query_vec = self.embedder.embed_text(query_text)
        query_vec = np.expand_dims(query_vec, axis=0)

        # Search top-k
        scores, indices = self.index.search(query_vec, min(top_k, len(self.jobs_df)))

        candidate_skills_set = set(s.strip().lower() for s in (candidate_skills or []))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            row = self.jobs_df.iloc[idx]
            job_skills_raw = str(row.get('skills', ''))
            job_skills_list = [s.strip() for s in job_skills_raw.split(',') if s.strip()]

            # Skill overlap analysis
            matching_skills = []
            missing_skills = []

            for js in job_skills_list:
                js_lower = js.lower()
                if any(cs == js_lower or cs in js_lower or js_lower in cs for cs in candidate_skills_set):
                    matching_skills.append(js)
                else:
                    missing_skills.append(js)

            result = JobMatchResult(
                job_id=row.get('job_id', idx),
                title=row.get('title', 'Unknown Title'),
                company=row.get('company', 'Unknown Company'),
                location=row.get('location', 'Remote'),
                experience_level=row.get('experience_level', 'Mid-level'),
                skills=job_skills_raw,
                description=row.get('description', ''),
                score=score,
                matching_skills=matching_skills,
                missing_skills=missing_skills
            )
            results.append(result)

        return results
