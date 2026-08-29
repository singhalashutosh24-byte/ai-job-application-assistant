import json
from langchain_ollama import ChatOllama


def analyze_jd(job_description: str) -> dict:
    llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.2)

    prompt = f"""You are a job description analyzer. Your task is to extract structured 
information from a job description and return it strictly as JSON.

Extract the following fields:
- required_skills: a list of objects, one per skill mentioned or implied 
  in the job description. Each object must have exactly two fields:
    - "skill": the name of the skill (string)
    - "importance": either "critical" or "nice_to_have"

IMPORTANT rules for required_skills:
1. Each "skill" value must be a SHORT, ATOMIC term — a single specific 
   technology, tool, or skill name only (e.g., "React", "PostgreSQL", 
   "AWS"). Do NOT copy a full sentence or requirement phrase as the 
   skill name.

   Example of WRONG output: 
   {{"skill": "Strong proficiency in JavaScript/TypeScript and React", "importance": "critical"}}

   Example of CORRECT output (break the sentence into separate atomic skills):
   {{"skill": "JavaScript", "importance": "critical"}},
   {{"skill": "TypeScript", "importance": "critical"}},
   {{"skill": "React", "importance": "critical"}}

2. Do NOT exclude optional or preferred skills from this list. If a 
   skill is mentioned with soft language like "a plus," "bonus," 
   "preferred," or "nice to have," you MUST still include it in 
   required_skills — just set its "importance" to "nice_to_have" 
   instead of leaving it out.

3. Do NOT include soft skills (like "communication skills"), 
   experience-duration requirements (like "4+ years of experience"), 
   or industry/domain background (like "fintech experience") in 
   required_skills. Domain/industry background goes in 
   domain_experience instead (see below).

4. Be careful with the word "or" — it has two different meanings:
   - If "or" connects two alternative options within a REQUIREMENT 
     (e.g., "PostgreSQL or MySQL", "AWS or GCP"), this means the 
     candidate needs AT LEAST ONE of them — both are still "critical."
   - Only mark something "nice_to_have" if it uses actual soft 
     language like "a plus," "bonus," "preferred" — never based on 
     the word "or" alone.

- job_role: the job title/role being hired for (as a single string)
- shortlist_criteria: a list of specific qualifications, experience 
  thresholds, or requirements used to shortlist candidates (as an 
  array of strings) — e.g. years of experience, degree requirements.
- domain_experience: a list of specific industry or domain backgrounds 
  mentioned in the job description (as an array of strings), e.g. 
  "fintech", "healthcare", "e-commerce", "payments". Include these 
  whether they are described as required OR merely preferred/bonus. 
  If no domain/industry background is mentioned at all, return an 
  empty list.

Respond with ONLY valid JSON in exactly this format, and nothing else 
— no explanations, no markdown code blocks, no extra commentary:

{{
  "required_skills": [
    {{"skill": "skill1", "importance": "critical"}},
    {{"skill": "skill2", "importance": "nice_to_have"}}
  ],
  "job_role": "role title",
  "shortlist_criteria": ["criteria1", "criteria2"],
  "domain_experience": ["domain1"]
}}

Job Description:
{job_description}
"""

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
        raise ValueError(f"JD Analyzer returned invalid JSON: {e}")

    return parsed_result


if __name__ == "__main__":
    sample_jd = """About Us: We're a fast-growing fintech startup on a mission to revolutionize digital payments across emerging markets. We're backed by top-tier VCs and have processed over $2B in transactions. Join our rocket ship!

The Role: We're looking for a rockstar Full Stack Developer to join our growing engineering team. You'll work closely with product and design to ship features fast in a high-growth environment.

What you'll do: Build and maintain scalable web applications, collaborate cross-functionally with product, design and data teams, write clean maintainable well-tested code, participate in code reviews and mentor junior engineers, own features end to end from design to deployment.

What we're looking for: 4+ years of professional software development experience. Strong proficiency in JavaScript/TypeScript and React. Experience with Node.js and building REST or GraphQL APIs. Solid understanding of relational databases (PostgreSQL or MySQL). Familiarity with cloud platforms, preferably AWS or GCP. Experience with CI/CD pipelines is a big plus. Prior fintech or payments experience is a bonus but not required. Excellent communication skills and a bias towards action. Bachelor's degree in CS, Engineering or equivalent practical experience.
"""

    result = analyze_jd(sample_jd)
    print(json.dumps(result, indent=2))