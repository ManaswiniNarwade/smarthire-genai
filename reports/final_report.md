# SmartHire GenAI — Capstone Project Final Report
**Executive Engineering Documentation & Architecture Walkthrough**

---

## 1. Project Overview & Problem Statement
In the modern hiring ecosystem, job seekers face a dual hurdle: resumes frequently fail automated Applicant Tracking Systems (ATS) due to poor formatting or missing keywords, while generic search engines fail to understand semantic alignments between candidate capabilities and job expectations. Furthermore, early-career candidates lack accessible, personalized, and domain-grounded mentoring to prepare for high-growth tech roles.

**SmartHire GenAI** addresses this problem by delivering an integrated, end-to-end Generative AI platform that:
1. **Parses Multi-Format Resumes:** Transforms unstructured PDF, DOCX, and TXT resumes into clean, validated JSON schemas with guaranteed field contracts.
2. **Performs Semantic Job Matching:** Leverages dense vector embeddings and a FAISS similarity search index to match candidates based on meaning and technical competencies rather than superficial keyword matches.
3. **Generates Actionable CV Improvements:** Provides targeted resume enhancements, tailored professional summaries, missing skill gap identification, and STAR / Google X-Y-Z bullet rewrites.
4. **Delivers Grounded AI Career Mentorship:** Deploys a RAG-based conversational agent grounded in industry job descriptions and curated career roadmaps to provide factual, hallucination-free guidance.
5. **Enforces Responsible AI Guardrails:** Intercepts prompt injections, prevents off-topic digressions, and validates input safety before LLM orchestration.

---

## 2. System Architecture & Component Design

The platform adopts a decoupled, modular architecture adhering to modern software engineering best practices:

```
[Resume Upload: PDF / DOCX / TXT]
            │
            ▼
[Document Loader & Sentence Splitter]
            │
            ▼
[LLM Structured Parser (Pydantic Schema)] ──► Clean JSON Profile
            │
            ├───────────────► [Dense Embedding (all-MiniLM-L6-v2)]
            │                         │
            │                         ▼
            │               [FAISS Vector Search Engine]
            │                         │
            │                         ▼
            │               Top-N Ranked Jobs & Skill Gap Analysis
            │
            ├───────────────► [CV Improvement Prompt Library]
            │                         │
            │                         ▼
            │               Tailored Summary + STAR Bullet Rewrites + ATS Score
            │
            └───────────────► [AI Career Mentor RAG Pipeline]
                                      │
                              [Guardrails Safety Check]
                                      │
                                      ▼
                            Grounded Guidance + Document Citations
                                      │
                                      ▼
                        [Streamlit Interactive Web Portal]
```

---

## 2.1 Project Directory Structure

```
smarthire-genai/
├── README.md                          # intro, setup, how to run, demo link
├── requirements.txt                   # langchain, faiss-cpu, streamlit, openai/anthropic, pypdf
├── .env.example                       # API key placeholders (never commit real keys)
├── .gitignore                         # ignore .env, data/, *.faiss, venv/
│
├── data/
│   ├── jobs/                          # job dataset (CSV)
│   ├── resumes/                       # sample resumes for testing
│   └── career_notes/                  # documents the mentor retrieves from
│
├── vectorstore/                       # saved FAISS index
│
├── notebooks/
│   ├── 01_embeddings_explore.ipynb    # test embeddings + similarity
│   ├── 02_build_faiss.ipynb           # build the job vector index
│   └── 03_rag_prototype.ipynb         # prototype the mentor RAG chain
│
├── src/
│   ├── config.py                      # model names, paths, params
│   ├── parsing/
│   │   ├── loader.py                  # load + chunk PDF/DOCX
│   │   └── resume_parser.py           # LLM structured output -> JSON
│   ├── search/
│   │   ├── embed.py                   # create embeddings
│   │   └── job_search.py              # FAISS index + top-N query
│   ├── generate/
│   │   ├── prompts.py                 # prompt library
│   │   └── cv_suggestions.py          # improvement generator
│   ├── mentor/
│   │   └── rag_chain.py               # RAG mentor with LangChain
│   ├── safety/
│   │   └── guardrails.py              # input validation + filters
│   └── evaluate.py                    # answer-quality checks
│
├── app/
│   └── streamlit_app.py               # the portal UI
│
└── reports/
    ├── answer_quality.md              # evaluation results
    └── final_report.pdf               # written project report
```

---

## 3. Module Breakdown & Implementation Details

### Module 1: Resume Parser (Structured Outputs)
- **Engine:** `src/parsing/resume_parser.py`
- **Methodology:** Ingests document streams via `pypdf`, `pdfplumber`, and `python-docx`. The raw text is passed to an LLM with strict JSON schema constraints.
- **Contract Enforcement:** Uses Pydantic v2 (`ParsedResume`, `ExperienceItem`, `EducationItem`) to guarantee type safety, contact extraction, standardized skills, and work history structuring.
- **Resilience:** Features an offline rule-based heuristic extractor with regex entity matching to ensure complete functionality in air-gapped or keyless environments.

### Module 2: Semantic Job Search (FAISS)
- **Engine:** `src/search/job_search.py`
- **Vector Space:** `sentence-transformers/all-MiniLM-L6-v2` producing 384-dimensional dense vectors normalized under L2 norm.
- **Indexing:** Employs `faiss.IndexFlatIP` (Inner Product on unit vectors), yielding exact cosine similarity without quantization loss.
- **Skill Analysis:** Dynamically intersects candidate skill sets with job requirement arrays, calculating match percentages, overlapping proficiencies, and missing technical prerequisites.

### Module 3: CV Improvement Generator
- **Engine:** `src/generate/cv_suggestions.py`
- **Prompt Architecture:** `src/generate/prompts.py`
- **Core Techniques:**
  - **Google X-Y-Z Formula:** "Accomplished [X], as measured by [Y], by doing [Z]".
  - **Active Verbs:** Replaces passive phrases with high-impact engineering leadership verbs.
  - **ATS Compatibility:** Computes an ATS score based on keyword coverage and structural compliance.

### Module 4: AI Career Mentor (RAG Pipeline)
- **Engine:** `src/mentor/rag_chain.py`
- **Knowledge Base:** Curated markdown guides (`data/career_notes/`) covering AI/ML roadmaps, Software Engineering, ATS preparation, and career transitions, alongside real-world job posting descriptions.
- **Orchestration:** LangChain LCEL architecture retrieving top-K semantic chunks and synthesizing grounded, actionable recommendations with transparent source attribution.

### Module 5: Guardrails Layer
- **Engine:** `src/safety/guardrails.py`
- **Defense-in-Depth:**
  1. Input sanitization (stripping non-printable control characters).
  2. Injection filter (regex rules detecting prompt overrides, jailbreaks, and system prompt extractors).
  3. Scope verification (domain term filtering + LLM classifier rejecting off-topic queries such as sports, politics, or entertainment).

### Module 6: Streamlit Web Portal
- **Engine:** `app/streamlit_app.py`
- **Design System:** Custom CSS featuring violet-purple gradients, glassmorphism cards, modern typography, responsive tabs, interactive badges, and an automated evaluation test runner.

---

## 4. Quantitative Evaluation & Quality Benchmarks

As documented in `reports/answer_quality.md`, the platform achieved:
- **Resume Parsing Entity Fidelity:** **100%** on target role, contact, and skills extraction.
- **Semantic Job Search Top-3 Precision:** **100%** across AI, Frontend, and DevOps test queries.
- **RAG Knowledge Base Grounding:** **100%** factual fidelity with zero unsupported hallucinations.
- **Guardrail Injection & Off-Topic Defense:** **100%** accuracy across security and domain boundaries.

---

## 5. Deployment Guide & Run Instructions
```bash
# 1. Clone repository
git clone https://github.com/your-username/smarthire-genai.git
cd smarthire-genai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys (optional; supports offline fallback mode)
cp .env.example .env

# 4. Run the Streamlit web application
streamlit run app/streamlit_app.py
```

---

## 6. Conclusion
SmartHire GenAI represents a production-ready Generative AI system demonstrating the practical synergy of LLM structured outputs, dense vector retrieval, prompt engineering, RAG orchestration, and responsible AI guardrails.
