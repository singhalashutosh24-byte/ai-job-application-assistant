import json
from langchain_ollama import ChatOllama

ATS_CRITIC_PROMPT = """You are an ATS (Applicant Tracking System) critic. Your task is to check 
how well a tailored resume covers the required skills for a job, using 
strict evidence-based checking - not assumptions.

You will be given:
- A list of required skills, each with an importance level (critical or nice_to_have)
- The candidate's tailored resume bullet points

Step 0: First, list every specific technology, tool, or skill that is 
literally written in the tailored bullet points below (word-for-word). 
This becomes your "evidence_list" - you must write this out explicitly 
as part of your JSON response, not just think about it internally. If 
the bullet points are empty, vague, or contain no technical content at 
all, your evidence_list should also be empty or very short.

Step 1: For each required skill, check if it appears in your OWN 
evidence_list from Step 0 (exact match or an unambiguous synonym, e.g. 
"Postgres" and "PostgreSQL" count as the same). You must only add a 
skill to "missing_skills" if it does NOT appear anywhere in the 
evidence_list you just wrote. Double-check: if a skill IS in your 
evidence_list, it must NOT also appear in missing_skills - that would 
be a contradiction. Likewise, if a skill is NOT in your evidence_list, 
it MUST appear in missing_skills - do not silently skip it.

Step 2: Write brief, actionable improvement_notes explaining what's 
genuinely missing.

Here is a worked example showing the correct behavior when a resume 
has little or no relevant content:

EXAMPLE INPUT:
Required Skills:
- Java (critical)
- Spring Boot (critical)
- MySQL (nice_to_have)

Tailored Resume Bullets:
Managed social media accounts and coordinated marketing campaigns.

EXAMPLE CORRECT OUTPUT:
{{
  "evidence_list": [],
  "missing_skills": [
    {{"skill": "Java", "importance": "critical"}},
    {{"skill": "Spring Boot", "importance": "critical"}},
    {{"skill": "MySQL", "importance": "nice_to_have"}}
  ],
  "improvement_notes": "The resume contains no technical content related to Java, Spring Boot, or MySQL. All required skills are missing."
}}

Notice: since the bullets contained NO technical skills at all, EVERY 
required skill was correctly marked as missing - none were skipped.

Now perform the same evidence-based analysis on the REAL input below.

Respond with ONLY valid JSON in exactly this format, and nothing else 
- no explanations, no markdown code blocks:

{{
  "evidence_list": ["skill_found_1", "skill_found_2"],
  "missing_skills": [
    {{"skill": "skill1", "importance": "critical"}}
  ],
  "improvement_notes": "..."
}}

Required Skills:
{required_skills}

Tailored Resume Bullets:
{tailored_bullets}
"""

CRITICAL_PENALTY = 15
NICE_TO_HAVE_PENALTY = 5
PASS_THRESHOLD = 70


def evaluate_resume(required_skills: list, tailored_bullets: list) -> dict:
    llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.1)

    skills_text = "\n".join([f"- {item['skill']} ({item['importance']})" for item in required_skills])
    bullets_text = "\n".join(tailored_bullets)

    prompt = ATS_CRITIC_PROMPT.format(
        required_skills=skills_text,
        tailored_bullets=bullets_text,
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
        raise ValueError(f"ATS Critic returned invalid JSON: {e}")

    # --- Score calculation happens HERE, in Python, not the LLM ---
    missing_skills = parsed_result.get("missing_skills", [])

    score = 100
    missing_keywords = []

    for item in missing_skills:
        skill_name = item.get("skill", "")
        importance = item.get("importance", "critical")  # default to critical if unclear

        if importance == "critical":
            score -= CRITICAL_PENALTY
        else:
            score -= NICE_TO_HAVE_PENALTY

        missing_keywords.append(skill_name)

    score = max(score, 0)  # never go below 0
    passed = score >= PASS_THRESHOLD

    return {
        "score": score,
        "passed": passed,
        "missing_keywords": missing_keywords,
        "improvement_notes": parsed_result.get("improvement_notes", ""),
    }


if __name__ == "__main__":
    sample_required_skills = [
        {"skill": "Python", "importance": "critical"},
        {"skill": "REST APIs", "importance": "critical"},
        {"skill": "PostgreSQL", "importance": "critical"},
        {"skill": "Docker", "importance": "nice_to_have"},
        {"skill": "AWS", "importance": "critical"},
    ]

    sample_tailored_bullets = [
        "Developed web applications using Python and Django",
        "Deployed applications on AWS EC2 instances",
    ]

    result = evaluate_resume(sample_required_skills, sample_tailored_bullets)
    print(json.dumps(result, indent=2))