from typing import TypedDict, List, Dict


class PipelineState(TypedDict):
    # Raw inputs
    resume_text: str
    job_description: str
    company_name: str

    # JD Analyzer outputs
    required_skills: List[Dict]
    job_role: str
    shortlist_criteria: List[str]
    domain_experience: List[str]

    # Resume Tailor outputs
    tailored_bullets: List[str]
    matched_skills: List[str]
    lacking_factors: List[str]

    # ATS Critic outputs
    score: int
    passed: bool
    missing_keywords: List[str]
    improvement_notes: str
    retry_count: int

    # Interview Prep outputs
    probable_questions: List[str]
    interview_process: List[str]
    interview_experiences: List[str]

#node wrapper functions
from agents.jd_analyzer import analyze_jd
from agents.resume_tailor import tailor_resume
from agents.ats_critic import evaluate_resume
from agents.interview_prep import get_interview_prep


def jd_analyzer_node(state: PipelineState) -> dict:
    result = analyze_jd(state["job_description"])
    return {
        "required_skills": result["required_skills"],
        "job_role": result["job_role"],
        "shortlist_criteria": result["shortlist_criteria"],
        "domain_experience": result["domain_experience"],
    }


def resume_tailor_node(state: PipelineState) -> dict:
    result = tailor_resume(
        resume_text=state["resume_text"],
        required_skills=state["required_skills"],
        job_role=state["job_role"],
        missing_keywords=state.get("missing_keywords"),
        improvement_notes=state.get("improvement_notes", ""),
    )
    
    
    
    return {
        "tailored_bullets": result["tailored_bullets"],
        "matched_skills": result["matched_skills"],
        "lacking_factors": result["lacking_factors"],
    }
   

def ats_critic_node(state: PipelineState) -> dict:
   
    
    result = evaluate_resume(
        required_skills=state["required_skills"],
        tailored_bullets=state["tailored_bullets"],
    )
    ...
    result = evaluate_resume(
        required_skills=state["required_skills"],
        tailored_bullets=state["tailored_bullets"],
    )
    current_retry_count = state.get("retry_count", 0)
    
   
    return {
        "score": result["score"],
        "passed": result["passed"],
        "missing_keywords": result["missing_keywords"],
        "improvement_notes": result["improvement_notes"],
        "retry_count": current_retry_count + 1,
    }
    


def interview_prep_node(state: PipelineState) -> dict:
    result = get_interview_prep(
        company_name=state["company_name"],
        job_role=state["job_role"],
    )
    return {
        "probable_questions": result["probable_questions"],
        "interview_process": result["interview_process"],
        "interview_experiences": result["interview_experiences"],
    }
from langgraph.graph import StateGraph, START, END

MAX_RETRIES = 3


def decide_next_step(state: PipelineState) -> str:
    if state["passed"] or state["retry_count"] >= MAX_RETRIES:
        return "proceed"
    else:
        return "retry"


# Graph to connect all agents 
builder = StateGraph(PipelineState)

# Register nodes
builder.add_node("jd_analyzer", jd_analyzer_node)
builder.add_node("resume_tailor", resume_tailor_node)
builder.add_node("ats_critic", ats_critic_node)
builder.add_node("interview_prep", interview_prep_node)

# Entry point: graph starts at JD Analyzer
builder.add_edge(START, "jd_analyzer")

# Fixed edges (no decision needed)
builder.add_edge("jd_analyzer", "resume_tailor")
builder.add_edge("resume_tailor", "ats_critic")

# Conditional edge: after ats_critic, decide whether to loop or proceed
builder.add_conditional_edges(
    "ats_critic",
    decide_next_step,
    {
        "retry": "resume_tailor",
        "proceed": "interview_prep",
    }
)

# Final edge: after interview_prep, the graph is done
builder.add_edge("interview_prep", END)

# Compile into a runnable app
app = builder.compile()

initial_state = {
        "resume_text": """
Raj Patel - Software Developer
Experience:
- Developed backend services using Python and Flask
- Wrote unit tests and participated in code reviews
- Used Git for version control and collaborated on GitHub
- Built internal tools to automate reporting tasks
""",
        "job_description": """
We are hiring a Backend Engineer with strong experience in Python, 
REST APIs, and PostgreSQL. Candidates should have at least 3 years 
of experience and familiarity with Docker and AWS.
""",
        "company_name": "Google",
        "retry_count": 0,
    }
final_state = app.invoke(initial_state)

print("\n\n=== FINAL STATE ===")
import json
print(json.dumps(final_state, indent=2))