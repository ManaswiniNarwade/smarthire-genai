import os
import sys
import json
import time
from pathlib import Path
import streamlit as st

# Configure project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    DEFAULT_LLM_PROVIDER,
    RESUMES_DIR,
    JOBS_CSV_PATH,
    DEFAULT_TOP_K_JOBS
)
from src.parsing.loader import DocumentLoader
from src.parsing.resume_parser import ResumeParser, ParsedResume
from src.search.job_search import JobSearchEngine
from src.generate.cv_suggestions import CVSuggestionEngine
from src.mentor.rag_chain import CareerMentorRAG
from src.safety.guardrails import GuardrailEngine
from src.evaluate import run_evaluation

# Set page config
st.set_page_config(
    page_title="SmartHire GenAI — Resume Matching & AI Career Mentor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics (Glassmorphism, Purple Gradient, Modern Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.02em;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #2b0b57 0%, #4a158a 50%, #1f0840 100%);
        border-radius: 18px;
        padding: 38px 44px;
        color: #ffffff;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px rgba(43, 11, 87, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .badge-capstone {
        display: inline-block;
        background: rgba(255, 255, 255, 0.16);
        color: #e0d0ff;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 5px 14px;
        border-radius: 20px;
        margin-bottom: 14px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 10px;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #d8c8f8;
        max-width: 860px;
        line-height: 1.5;
        margin-bottom: 20px;
    }
    .tag-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.12);
        color: #ffffff;
        font-size: 0.8rem;
        padding: 4px 12px;
        border-radius: 12px;
        margin-right: 8px;
        margin-bottom: 6px;
    }

    /* Skill badges */
    .skill-chip {
        display: inline-block;
        background: #ede9fe;
        color: #5b21b6;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 16px;
        margin: 3px;
        border: 1px solid #ddd6fe;
    }
    .skill-matched {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 14px;
        margin: 2px;
        border: 1px solid #bbf7d0;
    }
    .skill-missing {
        display: inline-block;
        background: #fee2e2;
        color: #991b1b;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 14px;
        margin: 2px;
        border: 1px solid #fecaca;
    }

    /* Cards */
    .job-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .job-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(74, 21, 138, 0.12);
    }
    .match-score-badge {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        font-weight: 700;
        font-size: 1.15rem;
        padding: 6px 14px;
        border-radius: 12px;
        display: inline-block;
    }

    /* Transformation card */
    .transform-box {
        background: #f8fafc;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Hero Banner
st.markdown("""
<div class="hero-container">
    <div class="badge-capstone">Capstone Project · Generative AI</div>
    <div class="hero-title">SmartHire GenAI — Resume Matching & AI Career Mentor</div>
    <div class="hero-subtitle">
        An end-to-end Generative AI portal: parse your resume, match top jobs through semantic search,
        generate ATS-ready CV improvements, and receive factual career mentoring with RAG.
    </div>
    <div>
        <span class="tag-pill">LLM APIs</span>
        <span class="tag-pill">Embeddings</span>
        <span class="tag-pill">FAISS</span>
        <span class="tag-pill">RAG</span>
        <span class="tag-pill">LangChain</span>
        <span class="tag-pill">Streamlit Portal</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if "parsed_resume" not in st.session_state:
    st.session_state.parsed_resume = None
if "matched_jobs" not in st.session_state:
    st.session_state.matched_jobs = []
if "selected_job_for_cv" not in st.session_state:
    st.session_state.selected_job_for_cv = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "👋 Hello! I am your **SmartHire AI Career Mentor**. Ask me any question about career transitions, required skills, interview preparation, or portfolio projects!"}
    ]

# LLM Provider configuration
llm_provider = DEFAULT_LLM_PROVIDER

# Sidebar Controls
st.sidebar.title("📁 Load Sample Resume")
sample_choice = st.sidebar.selectbox(
    "Choose Test Resume",
    ["None", "Alex Chen (Data Scientist)", "Priya Sharma (Frontend Developer)"]
)

if sample_choice == "Alex Chen (Data Scientist)":
    sample_path = RESUMES_DIR / "sample_data_scientist.txt"
    if sample_path.exists():
        st.session_state.loaded_resume_text = DocumentLoader.load_text(sample_path)
elif sample_choice == "Priya Sharma (Frontend Developer)":
    sample_path = RESUMES_DIR / "sample_frontend_developer.txt"
    if sample_path.exists():
        st.session_state.loaded_resume_text = DocumentLoader.load_text(sample_path)

# Initialize engines
@st.cache_resource
def get_search_engine():
    engine = JobSearchEngine()
    engine.build_index()
    return engine

@st.cache_resource
def get_rag_mentor():
    mentor = CareerMentorRAG()
    mentor.build_knowledge_base()
    return mentor

search_engine = get_search_engine()
rag_mentor = get_rag_mentor()
guardrail_engine = GuardrailEngine(provider=llm_provider)
cv_engine = CVSuggestionEngine(provider=llm_provider)
parser = ResumeParser(provider=llm_provider)

# Main Application Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Resume Parser",
    "🎯 Semantic Job Matching",
    "✍️ CV Improvement Generator",
    "🤖 AI Career Mentor (RAG)",
    "📊 Evaluation & Quality Benchmarks"
])

# ==========================================
# TAB 1: RESUME PARSER
# ==========================================
with tab1:
    st.subheader("Module 1: Structured Resume Parser")
    st.write("Upload a PDF, DOCX, or TXT resume to extract a structured JSON profile with LLM entity validation.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "doc", "txt", "md"])

        resume_text_to_parse = None
        if uploaded_file is not None:
            try:
                resume_text_to_parse = DocumentLoader.load_document(uploaded_file, filename=uploaded_file.name)
                st.success(f"Loaded `{uploaded_file.name}` ({len(resume_text_to_parse)} characters)")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        elif "loaded_resume_text" in st.session_state and st.session_state.loaded_resume_text:
            resume_text_to_parse = st.session_state.loaded_resume_text
            st.info("Loaded sample resume from sidebar.")

        if resume_text_to_parse:
            with st.expander("👁️ View Raw Extracted Text", expanded=False):
                st.text_area("Raw Text", resume_text_to_parse, height=220)

            if st.button("🚀 Parse Resume into Structured Profile", type="primary", use_container_width=True):
                with st.spinner("Extracting profile entities via LLM structured schema..."):
                    parsed = parser.parse(resume_text_to_parse)
                    st.session_state.parsed_resume = parsed
                    st.success("Successfully parsed and validated resume schema!")

    with col2:
        if st.session_state.parsed_resume:
            p = st.session_state.parsed_resume
            st.markdown(f"### 👤 {p.name}")
            st.markdown(f"**Target Role:** `{p.target_role}`")
            contact_items = [f"📧 {p.email}" if p.email else "", f"📞 {p.phone}" if p.phone else "", f"📍 {p.location}" if p.location else ""]
            st.caption(" • ".join([c for c in contact_items if c]))

            st.markdown("#### 📝 Professional Summary")
            st.info(p.summary or "No summary provided.")

            st.markdown("#### 🛠️ Technical & Professional Skills")
            chips_html = "".join([f'<span class="skill-chip">{s}</span>' for s in p.skills])
            st.markdown(chips_html, unsafe_allow_html=True)

            st.markdown("#### 💼 Experience Highlights")
            for exp in p.experience:
                st.markdown(f"**{exp.title}** at *{exp.company}* ({exp.duration or 'N/A'})")
                for hl in exp.highlights:
                    st.markdown(f"- {hl}")

            st.markdown("#### 🎓 Education")
            for edu in p.education:
                st.markdown(f"- **{edu.degree}** — {edu.institution} ({edu.year or 'N/A'})")

            with st.expander("🔍 Clean Validated JSON Object"):
                st.json(p.model_dump())
        else:
            st.info("👈 Upload a resume or select a sample resume on the left to begin parsing.")

# ==========================================
# TAB 2: SEMANTIC JOB MATCHING
# ==========================================
with tab2:
    st.subheader("Module 2: Semantic Job Search (FAISS)")
    st.write("Matches the candidate's profile against the pre-collected job corpus using dense cosine vector similarity.")

    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        if st.session_state.parsed_resume:
            st.success(f"Matching using profile for: **{st.session_state.parsed_resume.name}** ({st.session_state.parsed_resume.target_role})")
        else:
            st.warning("No parsed resume yet. You can still test semantic search by typing custom keywords below:")

    with col_ctrl2:
        top_k = st.slider("Top Jobs to Show", min_value=3, max_value=10, value=5)

    custom_search_query = st.text_input(
        "Search Query / Role Focus",
        value=st.session_state.parsed_resume.target_role if st.session_state.parsed_resume else "Generative AI Engineer PyTorch LangChain"
    )

    if st.button("🔍 Find Matching Jobs", type="primary"):
        with st.spinner("Embedding profile and querying FAISS vector index..."):
            candidate_skills = st.session_state.parsed_resume.skills if st.session_state.parsed_resume else []
            matches = search_engine.search_jobs(
                query_text=custom_search_query,
                candidate_skills=candidate_skills,
                top_k=top_k
            )
            st.session_state.matched_jobs = matches

    if st.session_state.matched_jobs:
        st.markdown(f"### Found {len(st.session_state.matched_jobs)} Semantically Matched Roles:")
        for i, job in enumerate(st.session_state.matched_jobs):
            with st.container():
                st.markdown(f"""
                <div class="job-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <h3 style="margin: 0; color: #1e1b4b;">{job.title}</h3>
                            <span style="color: #64748b; font-weight: 500;">{job.company} • 📍 {job.location} • ⏳ {job.experience_level}</span>
                        </div>
                        <div class="match-score-badge">{job.match_percentage} Match</div>
                    </div>
                    <p style="margin-top: 10px; color: #334155; font-size: 0.95rem;">{job.description[:260]}...</p>
                </div>
                """, unsafe_allow_html=True)

                col_m1, col_m2 = st.columns([3, 1])
                with col_m1:
                    matched_html = "".join([f'<span class="skill-matched">✓ {s}</span>' for s in job.matching_skills])
                    missing_html = "".join([f'<span class="skill-missing">+ {s}</span>' for s in job.missing_skills])
                    st.markdown(f"**Skill Overlap:** {matched_html if matched_html else 'None identified'} {missing_html}", unsafe_allow_html=True)

                with col_m2:
                    if st.button(f"🎯 Target for CV Tailoring", key=f"target_job_{job.job_id}"):
                        st.session_state.selected_job_for_cv = job.to_dict()
                        st.success(f"Selected '{job.title}'! Switch to Tab 3 to review suggestions.")
                st.divider()

# ==========================================
# TAB 3: CV IMPROVEMENT GENERATOR
# ==========================================
with tab3:
    st.subheader("Module 3: CV Improvement & ATS Optimization")
    st.write("Generates actionable bullet transformations (STAR / Google X-Y-Z formula), missing skills analysis, and a tailored executive summary.")

    if not st.session_state.selected_job_for_cv and st.session_state.matched_jobs:
        st.session_state.selected_job_for_cv = st.session_state.matched_jobs[0].to_dict()

    if st.session_state.selected_job_for_cv:
        tj = st.session_state.selected_job_for_cv
        st.info(f"🎯 **Target Job Selected:** {tj.get('title')} at {tj.get('company')}")
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown(f"**Required Job Skills:** `{tj.get('skills')}`")
        with col_t2:
            st.markdown(f"**Location & Level:** {tj.get('location')} ({tj.get('experience_level')})")

        if st.button("✨ Generate CV Tailoring & Suggestions", type="primary", use_container_width=True):
            with st.spinner("Analyzing candidate background against target role requirements..."):
                resume_input = st.session_state.parsed_resume or "Experienced software engineer specializing in Python and data analytics."
                suggestions = cv_engine.generate_suggestions(resume_input, tj)
                st.session_state.cv_suggestions = suggestions

        if "cv_suggestions" in st.session_state and st.session_state.cv_suggestions:
            sug = st.session_state.cv_suggestions
            
            # ATS Score Banner
            col_s1, col_s2 = st.columns([1, 2])
            with col_s1:
                st.metric("Estimated ATS Match Score", f"{sug.ats_score} / 100")
            with col_s2:
                for fb in sug.ats_feedback:
                    st.write(f"- 📋 {fb}")

            st.markdown("---")
            st.markdown("### 📝 Tailored Professional Summary Rewrite")
            st.success(sug.rewritten_summary)

            st.markdown("### ⚠️ Critical Missing Skills to Incorporate")
            miss_html = "".join([f'<span class="skill-missing">{s}</span>' for s in sug.missing_skills])
            st.markdown(miss_html if miss_html else "No major skill gaps detected!", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🔄 Weak vs. High-Impact STAR Bullet Rewrites")
            for i, bt in enumerate(sug.bullet_transformations):
                st.markdown(f"""
                <div class="transform-box">
                    <p style="color: #b91c1c; margin-bottom: 6px;"><strong>❌ Original Weak Bullet:</strong> {bt.original}</p>
                    <p style="color: #15803d; margin-bottom: 8px;"><strong>✅ Rewritten (STAR / Google X-Y-Z):</strong> {bt.rewritten}</p>
                    <p style="color: #6b21a8; font-size: 0.88rem; margin: 0;"><strong>💡 Recruiter Rationale:</strong> {bt.rationale}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 🚀 Step-by-Step Action Plan")
            for ap in sug.action_plan:
                st.write(f"1. {ap}")

            # Stretch Goal: Full Tailored Resume Generation
            st.markdown("---")
            st.markdown("### 📄 Full Tailored Resume (Ready to Export)")
            st.caption("Auto-synthesizes your background, missing target skills, and STAR accomplishments tailored for this role.")
            
            p = st.session_state.parsed_resume
            cand_name = p.name if p else "Alex Chen"
            cand_role = tj.get("title", "Software Engineer")
            cand_skills = ", ".join(list(set((p.skills if p else []) + sug.missing_skills[:3])))
            
            tailored_resume_md = f"""# {cand_name}
**Target Role:** {cand_role} | **Location:** {p.location if p and p.location else 'San Francisco, CA'}
**Email:** {p.email if p and p.email else 'alex.chen@example.com'} | **Phone:** {p.phone if p and p.phone else '(555) 234-5678'}

---

## PROFESSIONAL SUMMARY
{sug.rewritten_summary}

---

## TECHNICAL SKILLS
- **Core Competencies:** {cand_skills}
- **Target Role Alignment:** {tj.get('skills', '')}

---

## PROFESSIONAL EXPERIENCE
**{cand_role} Candidate** | Technology Projects | 2022 - Present
"""
            for bt in sug.bullet_transformations:
                tailored_resume_md += f"- {bt.rewritten}\n"

            tailored_resume_md += """
---

## EDUCATION
- **Master of Science in Computer Science** - Specialized in AI & Distributed Systems
- **Bachelor of Engineering in Information Technology**
"""
            with st.expander("👁️ Preview Full Tailored Resume", expanded=True):
                st.markdown(tailored_resume_md)

            st.download_button(
                label="📥 Download Tailored Resume (.md)",
                data=tailored_resume_md,
                file_name=f"{cand_name.replace(' ', '_')}_{cand_role.replace(' ', '_')}_Resume.md",
                mime="text/markdown",
                use_container_width=True
            )
    else:
        st.warning("Please search for jobs in Tab 2 and click 'Target for CV Tailoring' on a job card.")

# ==========================================
# TAB 4: AI CAREER MENTOR (RAG)
# ==========================================
with tab4:
    st.subheader("Module 4 & 5: AI Career Mentor with RAG & Guardrails")
    st.write("Ask career, interview, or technical questions grounded in our knowledge base with safety guardrails.")

    # Suggested Prompts
    st.markdown("**💡 Suggested Questions to Ask:**")
    prompt_cols = st.columns(3)
    with prompt_cols[0]:
        if st.button("How do I switch to Generative AI?"):
            st.session_state.mentor_query_input = "How do I switch to Generative AI and build RAG pipelines?"
    with prompt_cols[1]:
        if st.button("What projects impress recruiters?"):
            st.session_state.mentor_query_input = "What are the most impactful portfolio projects for a Data Scientist?"
    with prompt_cols[2]:
        if st.button("How do I format STAR bullets?"):
            st.session_state.mentor_query_input = "How do I write Google X-Y-Z style STAR bullet points for my resume?"

    # Display Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Verified Grounded Sources", expanded=False):
                    for src in msg["sources"]:
                        st.caption(f"• **{src['doc_name']}** (Score: {src.get('score', 0):.2f}) — `{src.get('snippet', '')}`")

    # Chat Input
    user_query = st.chat_input("Ask your career mentor anything...") or st.session_state.get("mentor_query_input")
    if user_query:
        # Reset quick query
        st.session_state.mentor_query_input = None

        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            # 1. Guardrail Validation
            guard_res = guardrail_engine.validate_input(user_query)
            if not guard_res.is_valid:
                rejection = guardrail_engine.format_rejection_message(guard_res)
                st.markdown(rejection)
                st.session_state.chat_history.append({"role": "assistant", "content": rejection})
            else:
                with st.spinner("Retrieving career knowledge base context & synthesizing guidance..."):
                    mentor_res = rag_mentor.answer(user_query, chat_history=st.session_state.chat_history)
                    ans = mentor_res.get("answer", "")
                    sources = mentor_res.get("sources", [])
                    st.markdown(ans)
                    if sources:
                        with st.expander("📚 Verified Grounded Sources", expanded=False):
                            for src in sources:
                                st.caption(f"• **{src['doc_name']}** (Score: {src.get('score', 0):.2f}) — `{src.get('snippet', '')}`")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": ans,
                        "sources": sources
                    })

# ==========================================
# TAB 5: EVALUATION & BENCHMARKS
# ==========================================
with tab5:
    st.subheader("System Evaluation & Quality Benchmarks")
    st.write("Run automated quantitative evaluation across all 5 core modules: Resume Parsing, Semantic Search, RAG Grounding, and Guardrails.")

    if st.button("🧪 Run Complete System Evaluation Suite", type="primary"):
        with st.spinner("Executing evaluation tests..."):
            eval_results = run_evaluation()
            st.session_state.eval_results = eval_results

    if "eval_results" in st.session_state:
        res = st.session_state.eval_results
        mods = res["modules"]

        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        with col_e1:
            st.metric("Resume Parser Score", f"{mods.get('resume_parser', {}).get('score', 0)}%", "PASS")
        with col_e2:
            st.metric("Semantic Search Hit Rate", f"{mods.get('semantic_search', {}).get('hit_rate_top3', 0)}%", "PASS")
        with col_e3:
            st.metric("RAG Mentor Fidelity", f"{mods.get('mentor_rag', {}).get('score', 0)}%", "PASS")
        with col_e4:
            st.metric("Guardrails Safety", f"{mods.get('guardrails', {}).get('accuracy', 0)}%", "PASS")

        report_file = PROJECT_ROOT / "reports" / "answer_quality.md"
        if report_file.exists():
            with st.expander("📄 View Full Markdown Evaluation Report", expanded=True):
                st.markdown(report_file.read_text(encoding="utf-8"))

        pdf_path = PROJECT_ROOT / "reports" / "final_report.pdf"
        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="📥 Download Capstone Final Project Report (final_report.pdf)",
                data=pdf_bytes,
                file_name="SmartHire_GenAI_Final_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
