import json
from langchain_ollama import ChatOllama

FIRST_ATTEMPT_PROMPT = """You are a resume tailoring assistant. Your task is to analyze how well a 
candidate's resume matches a job's required skills, and rewrite resume 
bullet points to better align with the job — without fabricating any 
experience the candidate doesn't actually have.

You will be given:
- The candidate's original resume text
- A list of required skills for the job
- The job role/title

Step 0: First, list every specific technology, tool, or skill that is 
literally written in the resume text below (word-for-word). Call this 
your "evidence list." Do not include anything not literally present 
in the text.

Step 1: For each required skill, check if it appears in your evidence 
list from Step 0 (exact match or an unambiguous synonym, e.g. "Postgres" 
and "PostgreSQL" count as the same). If it does not appear in your 
evidence list, it MUST go into lacking_factors, no exceptions.

Step 2: Rewrite the resume's bullet points to better emphasize the 
matched skills and align with the job role — using stronger action 
verbs and relevant keywords where truthful. Do not fabricate new 
achievements, numbers, or responsibilities. Only rephrase what is 
already true in the original resume.

Respond with ONLY valid JSON in exactly this format, and nothing else 
— no explanations, no markdown code blocks:

{{
  "matched_skills": ["skill1", "skill2"],
  "lacking_factors": ["skill3", "skill4"],
  "tailored_bullets": ["rewritten bullet 1", "rewritten bullet 2"]
}}

Job Role: {job_role}

Required Skills: {required_skills}

Original Resume:
{resume_text}
"""

RETRY_PROMPT = """You are a resume tailoring assistant. Your previous attempt to tailor 
this resume did not pass the ATS check. You must revise it using the 
specific feedback below — without fabricating any experience the 
candidate doesn't actually have.

Step 0: First, list every specific technology, tool, or skill that is 
literally written in the resume text below (word-for-word). Call this 
your "evidence list." Do not include anything not literally present 
in the text.

Step 1: For each required skill, check if it appears in your evidence 
list from Step 0 (exact match or an unambiguous synonym, e.g. "Postgres" 
and "PostgreSQL" count as the same). If it does not appear in your 
evidence list, it MUST go into lacking_factors, no exceptions.


Step 2: Pay special attention to these ATS keywords that were missing 
last time: {missing_keywords}
And this specific feedback: {improvement_notes}
Try to naturally incorporate any of these missing keywords ONLY IF 
they are truthfully supported by the candidate's actual experience. 
If they are not supported, keep them in lacking_factors instead of 
forcing them in.

Respond with ONLY valid JSON in exactly this format, and nothing else:

{{
  "matched_skills": ["skill1", "skill2"],
  "lacking_factors": ["skill3", "skill4"],
  "tailored_bullets": ["rewritten bullet 1", "rewritten bullet 2"]
}}

Job Role: {job_role}

Required Skills: {required_skills}

Original Resume:
{resume_text}
"""


def tailor_resume(resume_text: str, required_skills: list, job_role: str,
                   missing_keywords: list = None, improvement_notes: str = "") -> dict:
    llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.1)

    # required_skills now comes in as [{"skill": "Python", "importance": "critical"}, ...]
    # Resume Tailor doesn't care about importance, so extract just the names
    skill_names = [item["skill"] for item in required_skills]

    
    

    is_retry = missing_keywords is not None and len(missing_keywords) > 0

    if is_retry:
        prompt = RETRY_PROMPT.format(
            job_role=job_role,
            required_skills=skill_names,
            resume_text=resume_text,
            missing_keywords=missing_keywords,
            improvement_notes=improvement_notes,
        )
    else:
        prompt = FIRST_ATTEMPT_PROMPT.format(
            job_role=job_role,
            required_skills=skill_names,
            resume_text=resume_text,
        )
    response = llm.invoke(prompt)
    raw_text = response.content.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        parsed_result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from model output:")
        print(raw_text)
        raise ValueError(f"Resume Tailor returned invalid JSON: {e}")

    return parsed_result


if __name__ == "__main__":
    sample_resume = """
John Doe - Software Engineer
Experience:
- Developed web applications using Python and Django
- Worked with MySQL databases to manage application data
- Collaborated with a team of 5 engineers on a customer portal project
- Deployed applications on AWS EC2 instances
- Wrote unit tests to ensure code quality
"""

    sample_required_skills = [
        {"skill": "Python", "importance": "critical"},
        {"skill": "REST APIs", "importance": "critical"},
        {"skill": "PostgreSQL", "importance": "critical"},
        {"skill": "Docker", "importance": "nice_to_have"},
        {"skill": "AWS", "importance": "critical"},
    ]
    sample_job_role = "Backend Engineer"

    result = tailor_resume(sample_resume, sample_required_skills, sample_job_role)
    print(json.dumps(result, indent=2))