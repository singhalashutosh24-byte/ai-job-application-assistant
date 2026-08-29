import json
from langchain_ollama import ChatOllama
from agents.resume_retriever import get_similar_examples

FIRST_ATTEMPT_PROMPT = """You are a resume tailoring assistant. Your task is to analyze how well a 
candidate's resume matches a job's required skills, and rewrite resume 
bullet points to better align with the job - without fabricating any 
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

IMPORTANT: Do not infer or assume a skill is present just because 
it's commonly associated with another skill that IS mentioned. 
For example, if the resume mentions "AWS" but never explicitly 
mentions "Docker," do NOT count Docker as matched - even if AWS 
and Docker are often used together in real jobs. Only count a skill 
as matched if that exact skill (or a clear direct synonym) appears 
in the resume text itself.

Step 2: Rewrite the resume's bullet points to better emphasize the 
matched skills and align with the job role - using stronger action 
verbs and relevant keywords where truthful. Do not fabricate new 
achievements, numbers, or responsibilities. Only rephrase what is 
already true in the original resume.

Here are some examples of well-written resume bullets for similar roles, 
for STYLE AND PHRASING INSPIRATION ONLY:

{style_examples}

IMPORTANT: These examples are from other people's experience. Do NOT 
copy any specific achievement, number, or fact from these examples 
into the candidate's resume. Only use them to inform tone, structure, 
and how technical accomplishments are typically phrased. Every claim 
in the tailored bullets must still come from the candidate's OWN 
original resume text.

Respond with ONLY valid JSON in exactly this format, and nothing else 
- no explanations, no markdown code blocks:

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
specific feedback below - without fabricating any experience the 
candidate doesn't actually have.

Step 1: Compare the required skills against the resume text, same 
rules as before - only count a skill as "matched" if it's genuinely 
present or demonstrated in the resume.

Step 2: Pay special attention to these ATS keywords that were missing 
last time: {missing_keywords}
And this specific feedback: {improvement_notes}
Try to naturally incorporate any of these missing keywords ONLY IF 
they are truthfully supported by the candidate's actual experience. 
If they are not supported, keep them in lacking_factors instead of 
forcing them in.

Here are some examples of well-written resume bullets for similar roles, 
for STYLE AND PHRASING INSPIRATION ONLY:

{style_examples}

IMPORTANT: These examples are from other people's experience. Do NOT 
copy any specific achievement, number, or fact from these examples 
into the candidate's resume. Only use them to inform tone, structure, 
and how technical accomplishments are typically phrased. Every claim 
in the tailored bullets must still come from the candidate's OWN 
original resume text.

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
    llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.4)

    skill_names = [item["skill"] for item in required_skills]

    # --- RAG retrieval step ---
    query = f"{job_role} with experience in {', '.join(skill_names)}"
    examples = get_similar_examples(query, n_results=3)
    style_examples_text = "\n".join([f"- {ex}" for ex in examples])

    is_retry = missing_keywords is not None and len(missing_keywords) > 0

    if is_retry:
        prompt = RETRY_PROMPT.format(
            job_role=job_role,
            required_skills=skill_names,
            resume_text=resume_text,
            missing_keywords=missing_keywords,
            improvement_notes=improvement_notes,
            style_examples=style_examples_text,
        )
    else:
        prompt = FIRST_ATTEMPT_PROMPT.format(
            job_role=job_role,
            required_skills=skill_names,
            resume_text=resume_text,
            style_examples=style_examples_text,
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