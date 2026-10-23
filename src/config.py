import os
from pathlib import Path
from dotenv import load_dotenv

# Base Project Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Load .env if present
load_dotenv(PROJECT_ROOT / ".env")

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
JOBS_CSV_PATH = DATA_DIR / "jobs" / "jobs.csv"
RESUMES_DIR = DATA_DIR / "resumes"
CAREER_NOTES_DIR = DATA_DIR / "career_notes"

# Vectorstore Paths
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
JOBS_FAISS_INDEX = VECTORSTORE_DIR / "jobs_index.faiss"
JOBS_METADATA_PKL = VECTORSTORE_DIR / "jobs_metadata.pkl"
CAREER_FAISS_INDEX = VECTORSTORE_DIR / "career_index.faiss"
CAREER_METADATA_PKL = VECTORSTORE_DIR / "career_metadata.pkl"

# Ensure directories exist
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
RESUMES_DIR.mkdir(parents=True, exist_ok=True)
CAREER_NOTES_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "jobs").mkdir(parents=True, exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")).strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()

# Embedding Settings
# Default to local sentence-transformers (all-MiniLM-L6-v2) for reliable, zero-cost, offline execution
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers").lower()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384

# LLM Settings
# Providers: 'gemini', 'openai', 'anthropic', 'mock'
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini" if GEMINI_API_KEY else ("openai" if OPENAI_API_KEY else "mock"))
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash" if DEFAULT_LLM_PROVIDER == "gemini" else "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Search & Retrieval Settings
DEFAULT_TOP_K_JOBS = 5
DEFAULT_RAG_TOP_K = 3
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
