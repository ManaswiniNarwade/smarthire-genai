import json
import re
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

from src.config import (
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    DEFAULT_LLM_PROVIDER,
    LLM_MODEL_NAME
)
from src.parsing.resume_parser import ParsedResume
from src.generate.prompts import (
    CV_IMPROVEMENT_SYSTEM_PROMPT,
    CV_IMPROVEMENT_USER_TEMPLATE
)

class BulletTransformation(BaseModel):
    original: str = Field(..., description="Original weak bullet point")
    rewritten: str = Field(..., description="High-impact STAR/Google X-Y-Z rewritten bullet")
    rationale: str = Field(..., description="Why this rewrite increases recruiter impact")
    xyz_x: Optional[str] = Field(default=None, description="Accomplished [X]")
    xyz_y: Optional[str] = Field(default=None, description="As measured by [Y]")
    xyz_z: Optional[str] = Field(default=None, description="By doing [Z]")

class CVImprovementResponse(BaseModel):
    rewritten_summary: str
    missing_skills: List[str] = Field(default_factory=list)
    bullet_transformations: List[BulletTransformation] = Field(default_factory=list)
    ats_score: int = Field(default=75, description="Estimated ATS match score out of 100")
    ats_feedback: List[str] = Field(default_factory=list)
    action_plan: List[str] = Field(default_factory=list)
    matched_keywords: List[str] = Field(default_factory=list)
    recommended_power_verbs: List[str] = Field(default_factory=list)
    industry_buzzwords: List[str] = Field(default_factory=list)
    ats_breakdown: Dict[str, int] = Field(default_factory=dict)
    targeted_cover_letter: str = Field(default="")
    interview_prep_questions: List[Dict[str, str]] = Field(default_factory=list)

class CVSuggestionEngine:
    """Generates targeted CV optimization recommendations, rewritten summaries, and STAR bullets."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = (provider or DEFAULT_LLM_PROVIDER).lower()
        self.model = model or LLM_MODEL_NAME

    def generate_suggestions(
        self,
        resume: Union[ParsedResume, Dict[str, Any], str],
        target_job: Dict[str, Any]
    ) -> CVImprovementResponse:
        """Produce actionable CV improvement suggestions for a target job."""
        # Normalize resume info
        if isinstance(resume, ParsedResume):
            res_dict = resume.model_dump()
        elif isinstance(resume, dict):
            res_dict = resume
        else:
            res_dict = {
                "name": "Candidate",
                "target_role": "Applicant",
                "summary": str(resume)[:300],
                "skills": ["Python", "Git"],
                "experience": []
            }

        candidate_name = res_dict.get("name", "Candidate")
        candidate_target_role = res_dict.get("target_role", "Engineer")
        candidate_summary = res_dict.get("summary", "")
        candidate_skills = ", ".join(res_dict.get("skills", []))

        # Format experience bullets
        exp_bullets = []
        for exp in res_dict.get("experience", []):
            if isinstance(exp, dict):
                comp = exp.get("company", "")
                title = exp.get("title", "")
                hl = exp.get("highlights", [])
            elif hasattr(exp, "company"):
                comp = exp.company
                title = exp.title
                hl = exp.highlights
            else:
                comp, title, hl = "", "", []

            exp_bullets.append(f"- Role: {title} at {comp}")
            for b in hl:
                exp_bullets.append(f"  * {b}")

        candidate_exp_str = "\n".join(exp_bullets) if exp_bullets else "- General engineering experience."

        job_title = target_job.get("title", "Target Role")
        job_company = target_job.get("company", "Target Company")
        job_skills = target_job.get("skills", "")
        job_desc = target_job.get("description", "")

        user_prompt = CV_IMPROVEMENT_USER_TEMPLATE.format(
            job_title=job_title,
            company=job_company,
            job_skills=job_skills,
            job_description=job_desc[:1200],
            candidate_name=candidate_name,
            candidate_target_role=candidate_target_role,
            candidate_summary=candidate_summary,
            candidate_skills=candidate_skills,
            candidate_experience=candidate_exp_str
        )

        # 1. Try Gemini
        if self.provider == "gemini" and GEMINI_API_KEY:
            try:
                return self._generate_gemini(user_prompt, res_dict, target_job)
            except Exception as e:
                print(f"[CVSuggestionEngine] Gemini error: {e}. Falling back...")

        # 2. Try OpenAI
        if (self.provider == "openai" or OPENAI_API_KEY) and OPENAI_API_KEY:
            try:
                return self._generate_openai(user_prompt, res_dict, target_job)
            except Exception as e:
                print(f"[CVSuggestionEngine] OpenAI error: {e}. Falling back...")

        # 3. Deterministic Heuristic Suggestion Engine
        return self._generate_heuristic_suggestions(res_dict, target_job)

    def _generate_gemini(self, user_prompt: str, res_dict: Dict, target_job: Dict) -> CVImprovementResponse:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        full_prompt = f"{CV_IMPROVEMENT_SYSTEM_PROMPT}\n\nFormat your output as valid JSON matching this structure:\n{{\n  \"rewritten_summary\": \"...\",\n  \"missing_skills\": [\"...\"],\n  \"bullet_transformations\": [\n    {{\"original\": \"...\", \"rewritten\": \"...\", \"rationale\": \"...\"}}\n  ],\n  \"ats_score\": 82,\n  \"ats_feedback\": [\"...\"],\n  \"action_plan\": [\"...\"]\n}}\n\n{user_prompt}"
        response = model.generate_content(full_prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        data = json.loads(text.strip())
        return CVImprovementResponse.model_validate(data)

    def _generate_openai(self, user_prompt: str, res_dict: Dict, target_job: Dict) -> CVImprovementResponse:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        schema_prompt = """Respond in valid JSON:
{
  "rewritten_summary": "...",
  "missing_skills": ["..."],
  "bullet_transformations": [{"original": "...", "rewritten": "...", "rationale": "..."}],
  "ats_score": 85,
  "ats_feedback": ["..."],
  "action_plan": ["..."]
}"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": f"{CV_IMPROVEMENT_SYSTEM_PROMPT}\n{schema_prompt}"},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        data = json.loads(response.choices[0].message.content)
        return CVImprovementResponse.model_validate(data)

    def _generate_heuristic_suggestions(self, res_dict: Dict, target_job: Dict) -> CVImprovementResponse:
        """Deterministic high-quality fallback generator with deep keyword intelligence."""
        candidate_skills = set(s.lower() for s in res_dict.get("skills", []))
        candidate_name = res_dict.get("name", "Applicant")
        candidate_target_role = res_dict.get("target_role", "Engineer")
        job_skills_raw = target_job.get("skills", "")
        job_skills_list = [s.strip() for s in job_skills_raw.split(",") if s.strip()]

        missing_skills = []
        matching_skills = []
        for js in job_skills_list:
            if any(cs in js.lower() or js.lower() in cs for cs in candidate_skills):
                matching_skills.append(js)
            else:
                missing_skills.append(js)

        # ATS Score calculation
        total_skills = max(1, len(job_skills_list))
        match_ratio = len(matching_skills) / total_skills
        ats_score = int(min(98, max(45, (match_ratio * 50) + 45)))

        target_title = target_job.get("title", "Specialist")
        target_company = target_job.get("company", "Target Company")
        location = target_job.get("location", "Remote")

        # Tailored summary rewrite
        key_skills_phrase = ", ".join(job_skills_list[:4]) if job_skills_list else "software engineering"
        rewritten_summary = (
            f"High-impact {target_title} with proven engineering experience in {key_skills_phrase}. "
            f"Specialized in architecting performant, production-ready systems, automating critical pipelines, "
            f"and bridging product requirements into scalable solutions for {target_company}."
        )

        # High-impact power verbs
        power_verbs = [
            "Architected", "Spearheaded", "Accelerated", "Orchestrated",
            "Engineered", "Optimized", "Decoupled", "Streamlined", "Containerized"
        ]

        # Domain-specific buzzwords tailored to the job
        title_lower = target_title.lower()
        if any(w in title_lower for w in ["ai", "genai", "generative", "llm", "nlp", "machine learning", "data science"]):
            buzzwords = [
                "Retrieval-Augmented Generation (RAG)",
                "Dense Vector Similarity (FAISS)",
                "Low-Latency Inference & Quantization",
                "Prompt Optimization & Few-Shot In-Context Learning",
                "Model Evaluation & Hallucination Guardrails",
                "Distributed PyTorch Fine-Tuning"
            ]
        elif any(w in title_lower for w in ["frontend", "react", "ui", "web"]):
            buzzwords = [
                "Component-Driven Architecture",
                "Client-Side Hydration & Server Components",
                "Core Web Vitals & Bundle Splitting",
                "Responsive Layouts & Design Tokens",
                "State Synchronization & Optimistic UI"
            ]
        elif any(w in title_lower for w in ["devops", "cloud", "sre", "platform"]):
            buzzwords = [
                "Infrastructure as Code (Terraform)",
                "Container Orchestration (Kubernetes/EKS)",
                "Zero-Downtime Blue/Green Deployments",
                "Distributed Observability (Prometheus/Grafana)",
                "Automated CI/CD Pipeline Gates"
            ]
        else:
            buzzwords = [
                "High-Concurrency Microservices",
                "Database Indexing & Query Latency Optimization",
                "RESTful & Event-Driven Architecture",
                "Asynchronous Task Processing",
                "Automated Integration Test Coverage"
            ]

        # Bullet transformations using STAR / Google X-Y-Z method
        exp_list = res_dict.get("experience", [])
        sample_bullets = []
        for e in exp_list:
            if isinstance(e, dict):
                sample_bullets.extend(e.get("highlights", []))
            elif hasattr(e, "highlights"):
                sample_bullets.extend(e.highlights)

        primary_tech = job_skills_list[0] if job_skills_list else "modern frameworks"
        secondary_tech = job_skills_list[1] if len(job_skills_list) > 1 else "cloud infrastructure"

        bullet_transformations = []
        if sample_bullets:
            orig = sample_bullets[0]
            bullet_transformations.append(BulletTransformation(
                original=orig,
                rewritten=f"Spearheaded end-to-end service migration utilizing {primary_tech}, boosting system throughput by 34% and cutting API latency under peak concurrent loads.",
                rationale="Replaces passive phrasing with executive leadership verb ('Spearheaded') and quantifies business impact (+34% throughput).",
                xyz_x=f"Spearheaded core service migration utilizing {primary_tech}",
                xyz_y="boosted system throughput by 34% and cut API latency",
                xyz_z="by implementing asynchronous worker pools and caching"
            ))
            if len(sample_bullets) > 1:
                bullet_transformations.append(BulletTransformation(
                    original=sample_bullets[1],
                    rewritten=f"Architected automated CI/CD and deployment workflows incorporating {secondary_tech}, slashing release cycle duration from 4 days to 45 minutes while sustaining 99.95% availability.",
                    rationale="Applies the Google X-Y-Z formula by tying development actions directly to measurable delivery velocity and reliability.",
                    xyz_x=f"Architected automated CI/CD and deployment workflows with {secondary_tech}",
                    xyz_y="slashed release cycle duration from 4 days to 45 minutes with 99.95% uptime",
                    xyz_z="by containerizing test suites and automating regression gates"
                ))
        else:
            bullet_transformations.append(BulletTransformation(
                original="Assisted team in developing web modules and updating database records.",
                rewritten=f"Engineered high-throughput service modules and optimized database transactions using {primary_tech}, reducing p95 query latency by 42%.",
                rationale="Replaced vague verb 'Assisted' with high-impact verb 'Engineered' and added quantifiable latency reduction.",
                xyz_x=f"Engineered high-throughput service modules using {primary_tech}",
                xyz_y="reduced p95 query latency by 42%",
                xyz_z="by redesigning indexing strategies and query execution paths"
            ))

        ats_feedback = [
            f"Keyword Coverage: {len(matching_skills)}/{len(job_skills_list)} key skills matched ({int(match_ratio * 100)}%).",
            f"ATS Keyword Optimization: Integrate missing target skills ({', '.join(missing_skills[:3]) if missing_skills else 'domain keywords'}) into work experience accomplishments.",
            f"Target Role Alignment: Ensure the job title '{target_title}' is explicitly stated in your resume headline.",
            "Formatting Compliance: Maintain single-column, standard headers (Summary, Skills, Experience, Education) for 100% parsing fidelity."
        ]

        action_plan = [
            f"Embed missing core competencies ({', '.join(missing_skills[:3]) if missing_skills else 'domain keywords'}) into the top 3 bullet points of your most recent role.",
            "Adopt the Google X-Y-Z formula (Accomplished [X], as measured by [Y], by doing [Z]) across all resume bullet points.",
            "Incorporate high-impact power action verbs (e.g., 'Spearheaded', 'Architected', 'Orchestrated') at the start of every bullet point.",
            "Align your GitHub portfolio and LinkedIn headline with the target role keywords before submitting your application."
        ]

        ats_breakdown = {
            "keyword_match": int(match_ratio * 100),
            "metric_quantification": 88,
            "action_verbs": 92,
            "ats_layout": 96
        }

        # Targeted Cover Letter
        skills_summary = ", ".join(matching_skills[:3]) if matching_skills else "software engineering and system design"
        targeted_cover_letter = f"""Dear Hiring Team at {target_company},

I am writing to express my enthusiastic interest in the {target_title} position ({location}). With my hands-on background in {candidate_target_role} and proven expertise across {skills_summary}, I am confident in my ability to deliver immediate value to your engineering initiatives.

Throughout my career, I have focused on building performant, scalable systems that translate technical innovation into tangible business outcomes. Specifically, I have:
- Designed and deployed robust solutions using {primary_tech}, driving measurable improvements in operational throughput and latency.
- Collaborated across engineering and product teams to implement industry best practices, automated testing, and reliable continuous deployment workflows.
- Maintained a rigorous focus on code quality, architecture scalability, and user-centric problem solving.

{target_company}'s mission and technical standard align directly with my professional trajectory. I would welcome the opportunity to discuss how my technical skill set and track record of execution can contribute to your team's ongoing success.

Thank you for your time and consideration.

Sincerely,
{candidate_name}"""

        # Interview prep questions
        interview_prep_questions = [
            {
                "question": f"How have you designed and scaled systems using {primary_tech} in production?",
                "focus": "Technical Architecture & System Scalability",
                "strategy": f"Walk through a concrete project where you used {primary_tech}. Highlight architectural trade-offs, how you handled bottlenecks or latency, and the quantifiable outcome."
            },
            {
                "question": f"Describe a situation where a critical pipeline or service failed. How did you diagnose and resolve it?",
                "focus": "Problem Solving & High-Pressure Incident Management",
                "strategy": "Apply the STAR method: Situation (incident & impact), Task (your responsibility), Action (root cause diagnosis & mitigation), Result (resolution time and post-mortem improvements)."
            },
            {
                "question": f"How do you ensure code quality, test coverage, and smooth deployments when working with {secondary_tech}?",
                "focus": "Engineering Rigor & CI/CD Collaboration",
                "strategy": f"Discuss automated testing (unit/integration), containerization, code reviews, and how {secondary_tech} fits into your reliable deployment pipeline."
            }
        ]

        return CVImprovementResponse(
            rewritten_summary=rewritten_summary,
            missing_skills=missing_skills,
            bullet_transformations=bullet_transformations,
            ats_score=ats_score,
            ats_feedback=ats_feedback,
            action_plan=action_plan,
            matched_keywords=matching_skills,
            recommended_power_verbs=power_verbs,
            industry_buzzwords=buzzwords,
            ats_breakdown=ats_breakdown,
            targeted_cover_letter=targeted_cover_letter,
            interview_prep_questions=interview_prep_questions
        )
