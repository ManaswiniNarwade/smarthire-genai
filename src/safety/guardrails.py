import re
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from src.config import (
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    DEFAULT_LLM_PROVIDER
)
from src.generate.prompts import GUARDRAIL_CLASSIFIER_PROMPT

class GuardrailResult(BaseModel):
    is_valid: bool = Field(..., description="True if query is safe and in-scope for Career Mentor")
    category: str = Field(..., description="'career_query', 'off_topic', 'injection_attempt', 'toxic', 'empty'")
    reason: str = Field(..., description="Explanation of validation outcome")
    sanitized_input: str = Field(..., description="Input stripped of malicious or malformed characters")

class GuardrailEngine:
    """Security and scope guardrail layer for input validation and prompt injection defense."""

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"system\s+prompt",
        r"reveal\s+your\s+prompt",
        r"bypass\s+rules",
        r"dan\s+mode",
        r"jailbreak",
        r"pretend\s+you\s+are\s+not",
        r"you\s+are\s+now\s+an\s+unfiltered",
        r"execute\s+command",
        r"<script.*?>",
    ]

    ALLOWED_CAREER_KEYWORDS = [
        "resume", "cv", "job", "career", "interview", "salary", "promotion", "switch", "transition",
        "internship", "hiring", "recruiter", "ats", "portfolio", "project", "skills", "experience",
        "python", "sql", "java", "react", "next.js", "data", "ml", "ai", "engineer", "analyst",
        "developer", "devops", "cloud", "aws", "docker", "faiss", "rag", "langchain", "llm", "backend",
        "frontend", "system design", "leetcode", "dsa", "github", "linkedin", "work", "tech", "role"
    ]

    EXPLICIT_OFFTOPIC_KEYWORDS = [
        "recipe", "bake a cake", "weather tomorrow", "horoscope", "celebrity gossip", "sports score",
        "football match", "movie plot", "crypto pump", "buy bitcoin", "diet plan", "presidential election"
    ]

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or DEFAULT_LLM_PROVIDER).lower()

    def sanitize(self, text: str) -> str:
        """Strip control characters and excessive whitespaces."""
        if not text:
            return ""
        # Remove null bytes and non-printable control characters
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        return cleaned.strip()

    def validate_input(self, user_input: str) -> GuardrailResult:
        """
        Validate user input against injection attacks, toxicity, and off-topic domain queries.
        """
        sanitized = self.sanitize(user_input)

        # 1. Empty or trivial check
        if not sanitized or len(sanitized) < 3:
            return GuardrailResult(
                is_valid=False,
                category="empty",
                reason="Input is empty or too short to process.",
                sanitized_input=sanitized
            )

        lower_text = sanitized.lower()

        # 2. Prompt Injection / Jailbreak Detection (High Priority)
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, lower_text):
                return GuardrailResult(
                    is_valid=False,
                    category="injection_attempt",
                    reason="Security violation: Detected prompt injection or instruction override attempt.",
                    sanitized_input=sanitized
                )

        # 3. Explicit Off-Topic Detection
        for off in self.EXPLICIT_OFFTOPIC_KEYWORDS:
            if off in lower_text:
                return GuardrailResult(
                    is_valid=False,
                    category="off_topic",
                    reason=f"Topic out of scope: '{off}' is unrelated to tech careers, resumes, or job matching.",
                    sanitized_input=sanitized
                )

        # 4. Keyword Grounding Check (Fast path)
        has_career_term = any(k in lower_text for k in self.ALLOWED_CAREER_KEYWORDS)
        if has_career_term:
            return GuardrailResult(
                is_valid=True,
                category="career_query",
                reason="Input verified as relevant career or engineering topic.",
                sanitized_input=sanitized
            )

        # 5. If ambiguous, try LLM Guardrail if API key available
        if self.provider == "gemini" and GEMINI_API_KEY:
            try:
                return self._llm_guardrail_gemini(sanitized)
            except Exception:
                pass
        elif (self.provider == "openai" or OPENAI_API_KEY) and OPENAI_API_KEY:
            try:
                return self._llm_guardrail_openai(sanitized)
            except Exception:
                pass

        # If general conversational greeting like "hi", "hello", "help"
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "how are you", "help me"]
        if any(g == lower_text or lower_text.startswith(g) for g in greetings):
            return GuardrailResult(
                is_valid=True,
                category="career_query",
                reason="User greeting accepted.",
                sanitized_input=sanitized
            )

        # Default fallback: allow with caution if query seems technical/inquisitive
        if "?" in sanitized or len(sanitized.split()) >= 3:
            return GuardrailResult(
                is_valid=True,
                category="career_query",
                reason="Query accepted under permissive general career scope.",
                sanitized_input=sanitized
            )

        return GuardrailResult(
            is_valid=False,
            category="off_topic",
            reason="Please ask a question related to career planning, resume reviews, technical skills, or interviews.",
            sanitized_input=sanitized
        )

    def _llm_guardrail_gemini(self, text: str) -> GuardrailResult:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"{GUARDRAIL_CLASSIFIER_PROMPT}\n\nUSER INPUT: {text}"
        resp = model.generate_content(prompt)
        raw = resp.text.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        data = json.loads(raw.strip())
        return GuardrailResult(
            is_valid=bool(data.get("is_safe", True)),
            category=data.get("category", "career_query"),
            reason=data.get("reason", "LLM validation passed"),
            sanitized_input=text
        )

    def _llm_guardrail_openai(self, text: str) -> GuardrailResult:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": GUARDRAIL_CLASSIFIER_PROMPT},
                {"role": "user", "content": text}
            ],
            temperature=0.0
        )
        data = json.loads(resp.choices[0].message.content)
        return GuardrailResult(
            is_valid=bool(data.get("is_safe", True)),
            category=data.get("category", "career_query"),
            reason=data.get("reason", "LLM validation passed"),
            sanitized_input=text
        )

    @staticmethod
    def format_rejection_message(guardrail_res: GuardrailResult) -> str:
        """Produce a courteous rejection explaining scope limitations."""
        if guardrail_res.category == "injection_attempt":
            return (
                "🛡️ **Security Alert:** Your input contains disallowed system override instructions. "
                "SmartHire GenAI only processes legitimate career, resume, and job guidance questions."
            )
        elif guardrail_res.category == "off_topic":
            return (
                "🎯 **Scope Notice:** SmartHire AI Career Mentor is dedicated specifically to career navigation, "
                "resume improvements, technical skills, and interview preparation. "
                "Please ask a question related to your tech career, job matching, or portfolio development!"
            )
        else:
            return f"⚠️ **Input Error:** {guardrail_res.reason}"
