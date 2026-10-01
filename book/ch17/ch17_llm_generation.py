"""
Chapter 17: LLM Generation - Making TalentLens Explain Itself
Data Voyage - Building TalentLens

TalentLens milestone: add the generation layer. A user pastes their CV,
gets back not just ranked jobs but plain-English explanations of fit,
skill gaps, and which roles to apply to first.

Run (demo - no API key needed):
    python book/ch17/ch17_llm_generation.py

Run (live - requires GROQ_API_KEY or OPENAI_API_KEY):
    export GROQ_API_KEY=gsk_...   # or set in repo-root `.env` (loaded automatically)
    python book/ch17/ch17_llm_generation.py --live

Outputs:
    book/ch17/reports/figures/ch17_generation_pipeline.png
    book/ch17/reports/figures/ch17_latency_breakdown.png
    book/ch17/reports/figures/ch17_prompt_token_budget.png
    book/ch17/reports/sample_output.md
"""

from __future__ import annotations  # noqa: E402

import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_repo_dotenv() -> None:
    """Load repo-root `.env` if present (does not override existing env vars)."""
    try:
        from dotenv import load_dotenv  # noqa: E402
    except ImportError:
        return
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)


_THIS_DIR = Path(__file__).resolve().parent
SAVE_DPI = 300
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = (
    "DejaVu Sans"  # the seaborn style prefers Arial, which lacks the ₹ glyph
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    # Provider: "groq" | "openai" | "stub"
    provider: str = os.getenv("LLM_PROVIDER", "stub")
    # Hosted model names are retired every few months. Override without
    # editing code: GROQ_MODEL=... / OPENAI_MODEL=... (see console.groq.com/docs/models).
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float = 0.1  # low = consistent structured output
    max_tokens: int = 1200
    max_retries: int = 3
    max_jobs_in_prompt: int = 5
    max_job_tokens: int = 500  # truncate each job description to this
    max_cv_tokens: int = 600
    figures_dir: Path = _THIS_DIR / "reports" / "figures"
    reports_dir: Path = _THIS_DIR / "reports"


# ---------------------------------------------------------------------------
# Pydantic schemas (import-guarded)
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel as _BM  # noqa: E402
    from pydantic import Field as _F  # noqa: E402
    from pydantic import ValidationError as _VE  # noqa: E402

    class JobMatch(_BM):
        job_id: str
        title: str
        fit_assessment: str  # "STRONG MATCH" | "GOOD MATCH" | "PARTIAL MATCH" | "NO MATCH"
        why_it_fits: str
        skill_gap: str
        apply_recommendation: str  # "YES" | "MAYBE" | "NO"

    class MatchExplanation(_BM):
        cv_summary: str
        top_recommendation: str
        job_matches: list[JobMatch]

    class CVProfile(_BM):
        skills: list[str] = _F(default_factory=list)
        years_experience: int = 0
        current_role: str = ""
        raw_text: str = ""

    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False
    logger.warning("pydantic not installed — using dict-based validation fallback")


# ---------------------------------------------------------------------------
# LLM client - unified interface for Groq, OpenAI, and stub
# ---------------------------------------------------------------------------


class LLMClient:
    """Unified LLM client supporting Groq, OpenAI, and a deterministic stub.

    Swapping providers requires changing Config.provider only.
    In tests, the stub provides deterministic output without API calls.

    Args:
        cfg: Config with provider, model, temperature, max_tokens settings.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._client = None
        self._provider = cfg.provider
        self._model = self._resolve_model()

    def _resolve_model(self) -> str:
        if self._provider == "groq":
            return self.cfg.groq_model
        if self._provider == "openai":
            return self.cfg.openai_model
        return "stub"

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._provider == "groq":
            try:
                from groq import Groq  # noqa: E402

                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not set")
                self._client = Groq(api_key=api_key)
                return self._client
            except ImportError:
                raise ImportError("pip install groq to use Groq provider")
        if self._provider == "openai":
            try:
                import openai  # noqa: E402

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                self._client = openai.OpenAI(api_key=api_key)
                return self._client
            except ImportError:
                raise ImportError("pip install openai to use OpenAI provider")
        return None

    def complete(self, system: str, user: str) -> tuple[str, float]:
        """Send a chat completion request and return (content, latency_seconds).

        Args:
            system: System prompt defining the LLM's role and rules.
            user: User message containing the actual task.

        Returns:
            Tuple of (response_content: str, latency_seconds: float).

        Raises:
            RuntimeError: If all retries fail.
        """
        if self._provider == "stub":
            return self._stub_complete(system, user)

        client = self._get_client()
        last_error = None

        for attempt in range(self.cfg.max_retries):
            try:
                t0 = time.perf_counter()
                response = client.chat.completions.create(
                    model=self._model,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                latency = time.perf_counter() - t0
                content = response.choices[0].message.content
                logger.info(
                    f"LLM call: model={self._model} "
                    f"latency={latency:.2f}s "
                    f"tokens={response.usage.total_tokens}"
                )
                return content, latency
            except Exception as exc:
                last_error = exc
                wait = 2**attempt
                logger.warning(f"LLM call failed (attempt {attempt+1}): {exc}. Retrying in {wait}s")
                time.sleep(wait)

        raise RuntimeError(f"All {self.cfg.max_retries} LLM retries failed: {last_error}")

    def _stub_complete(self, system: str, user: str) -> tuple[str, float]:
        """Deterministic stub - same input always produces same output, no API needed.

        Uses a hash of the user message to select from pre-written responses,
        so the output is realistic and consistent across runs.

        Returns:
            Tuple of (json_string, simulated_latency).
        """
        import re as _re  # noqa: E402

        sim_latency = 0.2 + (len(user) / 10_000) * 0.3

        # Detect CV parsing call - system prompt asks for skills/years/current_role JSON
        if "years_experience" in system and "current_role" in system:
            skill_keywords = [
                "Python",
                "PyTorch",
                "TensorFlow",
                "RAG",
                "FastAPI",
                "NLP",
                "LLMs",
                "Docker",
                "PostgreSQL",
                "MLflow",
                "scikit-learn",
                "SQL",
                "Spark",
                "transformers",
                "pgvector",
            ]
            found = [s for s in skill_keywords if s.lower() in user.lower()]
            m_years = _re.search(r"(\d+)\s*(?:\+\s*)?(?:years?|yrs?)", user, _re.IGNORECASE)
            years = int(m_years.group(1)) if m_years else 3
            m_role = _re.search(
                r"(?:Lead|Senior|Principal|Staff|Engineer|Scientist)[^\n,]{0,30}",
                user,
                _re.IGNORECASE,
            )
            role = m_role.group(0).strip() if m_role else "AI/ML Engineer"
            return (
                json.dumps({"skills": found[:10], "years_experience": years, "current_role": role}),
                sim_latency,
            )

        # Explanation call
        job_ids = _extract_job_ids_from_prompt(user)[: self.cfg.max_jobs_in_prompt]
        cv_skills = _extract_skills_from_prompt(user)

        assessments = ["STRONG MATCH", "GOOD MATCH", "GOOD MATCH", "PARTIAL MATCH", "NO MATCH"]
        recommendations = ["YES", "YES", "YES", "MAYBE", "NO"]
        why_templates = [
            "Your {} background directly matches the core technical requirements.",
            "Strong alignment on {} skills. Role scope matches your experience level.",
            "Good technical overlap on {}. Consider this a stretch role.",
            "Limited {} overlap — role is adjacent but not a direct match.",
            "Significant skill mismatch. Role requires different core competencies.",
        ]
        gap_templates = [
            "Minor gaps only — worth addressing briefly in the cover letter.",
            "Some gaps in the required tech stack that could be bridged quickly.",
            "Moderate gaps — you'd need 2-3 months to fill the primary missing skills.",
            "Significant gaps in core requirements. Longer learning curve.",
            "Core requirement mismatch — this role would require major upskilling.",
        ]

        skill_str = ", ".join(cv_skills[:3]) if cv_skills else "Python, ML"
        matches = []
        for i, jid in enumerate(job_ids):
            idx = min(i, len(assessments) - 1)
            matches.append(
                {
                    "job_id": jid,
                    "title": f"Role {i+1}",
                    "fit_assessment": assessments[idx],
                    "why_it_fits": why_templates[idx].format(skill_str),
                    "skill_gap": gap_templates[idx],
                    "apply_recommendation": recommendations[idx],
                }
            )

        result = {
            "cv_summary": f"Candidate with {skill_str} background, ~{len(cv_skills)} core skills identified.",
            "top_recommendation": f"Prioritise applying to the top {min(2, len(matches))} matches this week.",
            "job_matches": matches,
        }
        return json.dumps(result), sim_latency


def _extract_job_ids_from_prompt(text: str) -> list[str]:
    """Extract job_id values from a prompt string for stub use."""
    import re  # noqa: E402

    ids = re.findall(r"Job ID:\s*(\S+)", text)
    if not ids:
        ids = re.findall(r'job_id["\s:]+(["\.\w]+)', text)
    return [i.strip("\"'") for i in ids][:10]


def _extract_skills_from_prompt(text: str) -> list[str]:
    """Extract skills list from a prompt string for stub use."""
    import re  # noqa: E402

    match = re.search(r'skills["\s:]+\[([^\]]+)\]', text, re.IGNORECASE)
    if match:
        return [s.strip().strip("\"'") for s in match.group(1).split(",")]
    return ["Python", "ML"]


# ---------------------------------------------------------------------------
# CV parser
# ---------------------------------------------------------------------------


class CVParser:
    """Extracts structured profile data from raw CV text using an LLM.

    Args:
        client: LLMClient instance.
        cfg: Config.
    """

    SYSTEM_PROMPT = """You are a CV parsing assistant.
Extract structured information from the provided CV text.
Output valid JSON only — no markdown, no preamble, no explanation.
Required format:
{
  "skills": ["list", "of", "technical", "skills"],
  "years_experience": <integer>,
  "current_role": "<most recent job title or empty string>"
}"""

    def __init__(self, client: LLMClient, cfg: Config) -> None:
        self.client = client
        self.cfg = cfg

    def parse(self, cv_text: str) -> "CVProfile":
        """Parse CV text into a structured CVProfile.

        Args:
            cv_text: Raw CV text (paste from PDF or typed profile).

        Returns:
            CVProfile with extracted skills, experience, and current role.
        """
        # Truncate to token budget (approximate: 4 chars per token)
        truncated = cv_text[: self.cfg.max_cv_tokens * 4]

        user_msg = f"<cv_text>\n{truncated}\n</cv_text>\n\nExtract the structured profile."
        raw, latency = self.client.complete(self.SYSTEM_PROMPT, user_msg)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"CV parse JSON failed, using empty profile. Raw: {raw[:100]}")
            data = {}

        if _PYDANTIC_AVAILABLE:
            return CVProfile(
                skills=data.get("skills", []),
                years_experience=int(data.get("years_experience", 0)),
                current_role=str(data.get("current_role", "")),
                raw_text=cv_text[:500],
            )
        return type(
            "CVProfile",
            (),
            {
                "skills": data.get("skills", []),
                "years_experience": int(data.get("years_experience", 0)),
                "current_role": str(data.get("current_role", "")),
                "raw_text": cv_text[:500],
            },
        )()


# ---------------------------------------------------------------------------
# Match explainer
# ---------------------------------------------------------------------------


class MatchExplainer:
    """Generates plain-English job match explanations using an LLM.

    Takes retrieved job results and a parsed CV, assembles a prompt,
    calls the LLM, and returns structured match explanations.

    Args:
        client: LLMClient instance.
        cfg: Config with max_jobs_in_prompt, max_job_tokens.
    """

    SYSTEM_PROMPT = """You are TalentLens, an AI career advisor with deep knowledge
of the tech job market in India and globally.

Your task: explain how well a candidate matches a list of job postings.

Rules:
- Base your analysis ONLY on the job postings and candidate profile provided.
  Do not invent requirements or skills not mentioned.
- Be specific: name the actual skills from the posting, not generic terms.
- Be honest: clearly state skill gaps when present.
- fit_assessment must be exactly one of: STRONG MATCH, GOOD MATCH, PARTIAL MATCH, NO MATCH
- apply_recommendation must be exactly one of: YES, MAYBE, NO
- Output valid JSON only. No markdown. No preamble. No explanation outside the JSON.

Required output format:
{
  "cv_summary": "<2-sentence summary of the candidate's profile>",
  "top_recommendation": "<which 1-2 jobs to prioritise and why, in plain English>",
  "job_matches": [
    {
      "job_id": "<id>",
      "title": "<job title>",
      "fit_assessment": "<STRONG MATCH|GOOD MATCH|PARTIAL MATCH|NO MATCH>",
      "why_it_fits": "<2-3 sentences: specific skills/experience that match>",
      "skill_gap": "<1-2 sentences: what's missing, or 'No significant gaps'>",
      "apply_recommendation": "<YES|MAYBE|NO>"
    }
  ]
}"""

    def __init__(self, client: LLMClient, cfg: Config) -> None:
        self.client = client
        self.cfg = cfg

    def explain(self, cv_profile, jobs: list[dict]) -> tuple[dict, float]:
        """Generate match explanations for a list of retrieved jobs.

        Args:
            cv_profile: CVProfile (or dict) with skills, years_experience, current_role.
            jobs: List of job dicts from vector search results.

        Returns:
            Tuple of (explanation_dict, latency_seconds).
        """
        top_jobs = jobs[: self.cfg.max_jobs_in_prompt]
        user_msg = self._build_prompt(cv_profile, top_jobs)
        raw, latency = self.client.complete(self.SYSTEM_PROMPT, user_msg)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            logger.error(f"Explanation JSON parse failed. Raw output: {raw[:200]}")
            result = self._fallback_explanation(top_jobs)

        # Validate with Pydantic if available
        if _PYDANTIC_AVAILABLE:
            try:
                validated = MatchExplanation.model_validate(result)
                return validated.model_dump(), latency
            except _VE as e:
                logger.warning(f"Explanation validation failed: {e}. Using raw dict.")

        return result, latency

    def _build_prompt(self, cv_profile, jobs: list[dict]) -> str:
        """Assemble the user message for the explanation request."""
        skills = getattr(
            cv_profile,
            "skills",
            cv_profile.get("skills", []) if isinstance(cv_profile, dict) else [],
        )
        years = getattr(cv_profile, "years_experience", 0)
        role = getattr(cv_profile, "current_role", "")

        parts = [
            "<candidate_profile>",
            f"Current role: {role or 'Not specified'}",
            f"Years of experience: {years}",
            f"Skills: {', '.join(skills) if skills else 'Not specified'}",
            "</candidate_profile>",
            "",
            "<job_postings>",
        ]

        for job in jobs:
            # Truncate description to token budget
            desc = str(job.get("chunk_text", job.get("skills", "")))
            desc_truncated = desc[: self.cfg.max_job_tokens * 4]  # ~4 chars/token
            parts.append(
                f"""
Job ID: {job.get("job_id", "unknown")}
Title: {job.get("title", "Unknown")}
Skills required: {job.get("skills", "")}
Remote: {job.get("is_remote", False)}
Salary: {_format_salary(job.get("salary_min"), job.get("salary_max"))}
Description: {desc_truncated}
---"""
            )

        parts.append("</job_postings>")
        parts.append("")
        parts.append("Analyse candidate fit for each job posting above.")
        return "\n".join(parts)

    def _fallback_explanation(self, jobs: list[dict]) -> dict:
        """Return minimal explanation when LLM call fails."""
        return {
            "cv_summary": "Profile analysis unavailable (LLM error).",
            "top_recommendation": "Review the top-ranked results based on similarity scores.",
            "job_matches": [
                {
                    "job_id": j.get("job_id", ""),
                    "title": j.get("title", ""),
                    "fit_assessment": "GOOD MATCH",
                    "why_it_fits": "Semantic similarity suggests relevant background.",
                    "skill_gap": "Detailed analysis unavailable.",
                    "apply_recommendation": "MAYBE",
                }
                for j in jobs
            ],
        }


def _format_salary(sal_min, sal_max) -> str:
    if sal_min and sal_max:
        return f"₹{sal_min/100_000:.0f}L–₹{sal_max/100_000:.0f}L"
    return "Not disclosed"


# ---------------------------------------------------------------------------
# TalentLens advisor - full pipeline
# ---------------------------------------------------------------------------


class TalentLensAdvisor:
    """End-to-end career advisor: CV text → structured match explanations.

    Wires together CVParser, vector search (stub for standalone use),
    and MatchExplainer into a single callable interface.

    Args:
        cfg: Config.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.llm = LLMClient(cfg)
        self.parser = CVParser(self.llm, cfg)
        self.explainer = MatchExplainer(self.llm, cfg)

    def advise(self, cv_text: str, jobs: list[dict]) -> dict:
        """Generate career advice from CV text and a list of job candidates.

        Args:
            cv_text: Raw CV text from the user.
            jobs: Pre-retrieved job candidates (from Ch17 vector search).

        Returns:
            Dict with cv_profile, explanation, and latency breakdown.
        """
        timings: dict[str, float] = {}

        # Step 1: Parse CV
        t0 = time.perf_counter()
        cv_profile = self.parser.parse(cv_text)
        timings["cv_parsing"] = time.perf_counter() - t0

        # Step 2: Generate explanations
        t0 = time.perf_counter()
        explanation, llm_latency = self.explainer.explain(cv_profile, jobs)
        timings["explanation"] = time.perf_counter() - t0
        timings["llm_latency"] = llm_latency

        return {
            "cv_profile": {
                "skills": getattr(cv_profile, "skills", []),
                "years_experience": getattr(cv_profile, "years_experience", 0),
                "current_role": getattr(cv_profile, "current_role", ""),
            },
            "explanation": explanation,
            "provider": self.cfg.provider,
            "model": self.llm._model,
            "timings": timings,
        }


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

DEMO_CV = """
Irfan Ali — AI Engineer
7 years experience in production AI/ML systems

Skills: Python, PyTorch, RAG pipelines, FastAPI, pgvector, LLMs,
NLP, transformers, scikit-learn, Docker, PostgreSQL, MLflow

Experience:
- Founder / AI Engineer at DataCortex IQ (2024–present)
- AI Engineer at Luminous (Schneider Electric) — NLP systems
- AI Engineer at Kuration AI (Hong Kong) — LLM products

Education: Professional Master's, Data Science & AI, IISER Tirupati

Published 11 Python libraries on PyPI including RAGNav, ragfallback,
AgentEnsemble. Two peer-reviewed papers on NLP/ML in 2025.
"""

DEMO_JOBS = [
    {
        "job_id": "job_001",
        "title": "Senior NLP Engineer",
        "skills": "Python|PyTorch|NLP|RAG|FastAPI|Transformers|pgvector",
        "is_remote": True,
        "salary_min": 2_800_000,
        "salary_max": 3_500_000,
        "similarity": 0.847,
        "chunk_text": (
            "We are building the next generation of AI-powered search. "
            "You will design and deploy RAG pipelines, work with vector databases (pgvector, Qdrant), "
            "and ship NLP models to production. Strong Python and transformer model experience required. "
            "FastAPI for API development. RLHF experience a plus."
        ),
    },
    {
        "job_id": "job_002",
        "title": "ML Engineer — Recommendation Systems",
        "skills": "Python|scikit-learn|Spark|MLflow|XGBoost|Airflow",
        "is_remote": False,
        "salary_min": 2_200_000,
        "salary_max": 3_000_000,
        "similarity": 0.791,
        "chunk_text": (
            "Build and maintain recommendation and ranking systems for a large e-commerce platform. "
            "You will work with Spark for large-scale feature engineering, XGBoost and LightGBM for "
            "model training, MLflow for experiment tracking, and Airflow for pipeline orchestration. "
            "Python expert, SQL required. Spark strongly preferred."
        ),
    },
    {
        "job_id": "job_003",
        "title": "AI Engineer — LLM Products",
        "skills": "Python|LLMs|FastAPI|pgvector|LangChain|OpenAI|Groq",
        "is_remote": True,
        "salary_min": 3_000_000,
        "salary_max": 4_200_000,
        "similarity": 0.838,
        "chunk_text": (
            "Build production LLM applications: document Q&A, chatbots, AI agents. "
            "You will work with OpenAI and Groq APIs, LangChain and LlamaIndex frameworks, "
            "and pgvector for semantic search. FastAPI for serving. "
            "Solid Python fundamentals. Experience shipping real LLM products required."
        ),
    },
    {
        "job_id": "job_004",
        "title": "Research Scientist — NLP",
        "skills": "Python|PyTorch|Research|NLP|Transformers|RLHF|Publications",
        "is_remote": False,
        "salary_min": 3_500_000,
        "salary_max": 6_000_000,
        "similarity": 0.762,
        "chunk_text": (
            "Conduct research on large language models. Publish papers at top venues (ACL, NeurIPS). "
            "Implement state-of-the-art architectures. RLHF and alignment research experience strongly preferred. "
            "PhD in ML, NLP, or related field preferred. Strong PyTorch and academic writing skills."
        ),
    },
    {
        "job_id": "job_005",
        "title": "Data Scientist — Analytics",
        "skills": "Python|SQL|pandas|Statistics|Tableau|Excel",
        "is_remote": False,
        "salary_min": 1_400_000,
        "salary_max": 2_000_000,
        "similarity": 0.612,
        "chunk_text": (
            "Analyse user behaviour data to drive product and business decisions. "
            "Build dashboards in Tableau, write SQL queries, run A/B tests, "
            "present insights to non-technical stakeholders. "
            "Python and pandas for data manipulation. Statistics required. ML background a plus."
        ),
    },
]


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def plot_generation_pipeline(cfg: Config) -> Path:
    """Architecture diagram: CV + search results → LLM → structured advice."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, label, sub="", color="#2196F3"):
        rect = plt.Rectangle(
            (x, y),
            w,
            h,
            facecolor=color,
            alpha=0.18,
            edgecolor=color,
            linewidth=1.8,
            zorder=2,
            clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2 + (0.16 if sub else 0),
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            zorder=3,
        )
        if sub:
            ax.text(
                x + w / 2,
                y + h / 2 - 0.22,
                sub,
                ha="center",
                va="center",
                fontsize=8,
                color="#555555",
                zorder=3,
            )

    def arr(x1, y1, x2, y2, label=""):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.8),
        )
        if label:
            ax.text(
                (x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center", fontsize=8, color="#555555"
            )

    # Inputs
    box(0.2, 3.8, 1.8, 1.0, "CV text", "raw paste", "#9C27B0")
    box(0.2, 2.2, 1.8, 1.0, "Job results", "Ch17 search", "#607D8B")

    # CV Parser
    arr(2.0, 4.3, 2.8, 4.3)
    box(2.8, 3.8, 2.0, 1.0, "CV Parser", "LLM call #1", "#FF9800")

    # Structured profile
    arr(4.8, 4.3, 5.6, 4.3)
    box(5.6, 3.8, 2.0, 1.0, "CV Profile", "skills, exp, role", "#4CAF50")

    # Combine
    arr(7.6, 4.3, 8.2, 3.5)
    arr(2.0, 2.7, 8.2, 3.2)

    # Prompt builder
    box(8.2, 2.8, 1.8, 1.2, "Prompt", "builder", "#FF9800")

    # LLM call
    arr(10.0, 3.4, 10.8, 3.4)
    box(10.8, 2.8, 1.8, 1.2, "LLM", "Groq / OpenAI\nJSON mode", "#2196F3")

    # Output
    arr(12.6, 3.4, 13.0, 3.4)
    box(13.0, 2.5, 0.8, 1.8, "Advice", "fit\ngaps\napply?", "#4CAF50")

    # Token budget annotation
    ax.annotate(
        "",
        xy=(10.0, 1.8),
        xytext=(8.2, 1.8),
        arrowprops=dict(arrowstyle="<->", color="#F44336", lw=1.5),
    )
    ax.text(
        9.1, 1.6, "~4,200 tokens\n(system + CV + 5 jobs)", ha="center", fontsize=8, color="#F44336"
    )

    # Retry annotation
    ax.annotate(
        "",
        xy=(10.8, 2.4),
        xytext=(10.8, 1.5),
        arrowprops=dict(arrowstyle="->", color="#607D8B", lw=1.5, linestyle="dashed"),
    )
    ax.text(10.8, 1.35, "retry (max 3)", ha="center", fontsize=8, color="#607D8B")

    ax.set_title(
        "TalentLens Generation Pipeline — CV → LLM → Structured Advice",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    legend_elements = [
        mpatches.Patch(facecolor="#9C27B0", alpha=0.25, label="User input"),
        mpatches.Patch(facecolor="#FF9800", alpha=0.25, label="LLM calls"),
        mpatches.Patch(facecolor="#4CAF50", alpha=0.25, label="Structured output"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    plt.tight_layout()
    out = cfg.figures_dir / "ch17_generation_pipeline.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def plot_latency_breakdown(timings_list: list[dict], cfg: Config) -> Path:
    """Bar chart showing latency breakdown across pipeline steps."""
    labels = [
        "CV parsing\n(LLM call #1)",
        "Vector search\n(Ch16, typical)",
        "Explanation\n(LLM call #2)",
        "Total",
    ]
    cv_t = np.mean([t.get("cv_parsing", 0.28) for t in timings_list]) * 1000
    search_t = 40.0  # typical in-memory search on the bundled index (Chapter 16), not measured here
    exp_t = np.mean([t.get("explanation", 0.34) for t in timings_list]) * 1000
    total = cv_t + search_t + exp_t

    values = [cv_t, search_t, exp_t, total]
    colors = ["#FF9800", "#4CAF50", "#2196F3", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", alpha=0.85, width=0.55)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{val:.0f}ms",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.axhline(1000, color="#F44336", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(-0.4, 1060, "1 s target", fontsize=9, color="#F44336", clip_on=True)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    source = (
        "stub provider — no real LLM latency; run with --live to measure"
        if cfg.provider == "stub"
        else f"measured: {cfg.provider}"
    )
    ax.set_title(
        f"TalentLens End-to-End Latency — CV to Structured Advice\n({source})",
        fontsize=13,
        fontweight="bold",
    )
    # Keep the 1 s target line inside the axes: a label drawn above the y-limit
    # makes bbox_inches="tight" stretch the saved figure to reach it.
    ax.set_ylim(0, max(max(values) * 1.25, 1150))
    out = cfg.figures_dir / "ch17_latency_breakdown.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


def plot_token_budget(cfg: Config) -> Path:
    """Stacked bar showing token budget breakdown per prompt."""
    components = {
        "System prompt": 320,
        "CV profile": 420,
        "5 job descriptions": 2500,
        "Instructions": 180,
        "Expected response": 800,
    }
    colors = ["#607D8B", "#9C27B0", "#2196F3", "#FF9800", "#4CAF50"]
    total = sum(components.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    left = 0
    for (label, tokens), color in zip(components.items(), colors):
        pct = tokens / total * 100
        ax.barh(
            0,
            tokens,
            left=left,
            color=color,
            edgecolor="white",
            height=0.5,
            label=f"{label} ({tokens:,} tok, {pct:.0f}%)",
        )
        ax.text(
            left + tokens / 2,
            0,
            f"{tokens:,}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if tokens > 400 else "black",
            fontweight="bold",
        )
        left += tokens

    # Context limit line
    ax.axvline(8192, color="#F44336", linewidth=2, linestyle="--")
    ax.text(8250, 0.3, "8K context\nlimit (Llama 3.1)", fontsize=8.5, color="#F44336")

    ax.set_xlim(0, 9000)
    ax.set_yticks([])
    ax.set_xlabel("Token count", fontsize=12)
    ax.set_title(
        f"Prompt Token Budget — Total: {total:,} / 8,192 token context window",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = cfg.figures_dir / "ch17_prompt_token_budget.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------


def run_demo(cfg: Config) -> dict:
    """Run the full advisor pipeline on demo data and print results."""
    advisor = TalentLensAdvisor(cfg)
    result = advisor.advise(DEMO_CV, DEMO_JOBS)

    _print_block(
        "TALENTLENS ADVISOR — FULL PIPELINE OUTPUT",
        [
            f"Provider:   {result['provider']} ({result['model']})",
            f"CV parsing: {result['timings']['cv_parsing']*1000:.0f}ms",
            f"Explanation:{result['timings']['explanation']*1000:.0f}ms",
            "",
            "CV PROFILE",
            f"  Skills:          {', '.join(result['cv_profile']['skills'][:6])}",
            f"  Experience:      {result['cv_profile']['years_experience']} years",
            f"  Current role:    {result['cv_profile']['current_role']}",
        ],
    )

    expl = result["explanation"]
    _print_block(
        "MATCH EXPLANATIONS",
        [
            f"  Summary: {expl.get('cv_summary', '')}",
            f"  Top rec: {expl.get('top_recommendation', '')}",
            "",
        ]
        + [
            f"  [{m.get('apply_recommendation','?')}] {m.get('title','')} — {m.get('fit_assessment','')}\n"
            f"       Why: {m.get('why_it_fits','')}\n"
            f"       Gap: {m.get('skill_gap','')}"
            for m in expl.get("job_matches", [])
        ],
    )

    return result


def write_sample_output(result: dict, cfg: Config) -> Path:
    """Write a markdown sample output file."""
    expl = result["explanation"]
    matches = expl.get("job_matches", [])

    lines = [
        "# TalentLens — Sample Generation Output",
        f"*Provider: {result['provider']} ({result['model']})*",
        "",
        "## CV Summary",
        expl.get("cv_summary", ""),
        "",
        "## Top Recommendation",
        expl.get("top_recommendation", ""),
        "",
        "## Job Match Analysis",
        "",
    ]
    for m in matches:
        lines += [
            f"### {m.get('title', '')} `{m.get('fit_assessment', '')}`",
            f"**Apply?** {m.get('apply_recommendation', '')}",
            "",
            f"**Why it fits:** {m.get('why_it_fits', '')}",
            "",
            f"**Skill gap:** {m.get('skill_gap', '')}",
            "",
            "---",
            "",
        ]
    lines += [
        "## Timing",
        f"- CV parsing: {result['timings']['cv_parsing']*1000:.0f}ms",
        f"- Explanation: {result['timings']['explanation']*1000:.0f}ms",
        f"- Total: {(result['timings']['cv_parsing'] + result['timings']['explanation'])*1000:.0f}ms",
    ]

    out = cfg.reports_dir / "sample_output.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _print_block(title: str, lines: list[str]) -> None:
    sep = "=" * 60
    logger.info(f"\n{sep}\n  {title}\n{sep}")
    for line in lines:
        logger.info(f"  {line}")


def _ensure_dirs(cfg: Config) -> None:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _load_repo_dotenv()
    live_mode = "--live" in sys.argv

    cfg = Config()
    if live_mode:
        # An explicit LLM_PROVIDER wins; otherwise try Groq, then OpenAI.
        explicit = os.getenv("LLM_PROVIDER", "").lower()
        if explicit in ("groq", "openai"):
            cfg.provider = explicit
        elif os.getenv("GROQ_API_KEY"):
            cfg.provider = "groq"
        elif os.getenv("OPENAI_API_KEY"):
            cfg.provider = "openai"
        else:
            logger.warning("No API key found. Set GROQ_API_KEY or OPENAI_API_KEY for live mode.")
            logger.warning("Falling back to stub mode.")
            cfg.provider = "stub"

    _ensure_dirs(cfg)

    logger.info("=" * 60)
    logger.info("  CHAPTER 17: LLM GENERATION LAYER")
    logger.info(f"  Provider: {cfg.provider} | Mode: {'live' if live_mode else 'demo (stub)'}")
    logger.info("=" * 60)

    logger.info("\n[1/5] Running generation pipeline...")
    result = run_demo(cfg)

    logger.info("\n[2/5] Writing sample output...")
    write_sample_output(result, cfg)

    logger.info("\n[3/5] Generation pipeline diagram...")
    plot_generation_pipeline(cfg)

    logger.info("\n[4/5] Latency breakdown chart...")
    plot_latency_breakdown([result["timings"]], cfg)

    logger.info("\n[5/5] Token budget diagram...")
    plot_token_budget(cfg)

    logger.info("\n" + "=" * 60)
    logger.info("  CHAPTER 17 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Output:  {cfg.reports_dir}/sample_output.md")
    logger.info(f"  Figures: {cfg.figures_dir}/")
    logger.info("\n  To run with a real LLM:")
    logger.info("    export GROQ_API_KEY=gsk_...")
    logger.info("    python book/ch17/ch17_llm_generation.py --live")
    logger.info("\nNext: Chapter 18 — Prompt engineering depth.")
    logger.info("Structured output validation, chain-of-thought, evaluation.")


if __name__ == "__main__":
    main()
