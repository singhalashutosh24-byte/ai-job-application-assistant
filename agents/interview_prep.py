# agents/interview_prep.py

import json
from langchain_ollama import ChatOllama
from agents.search_provider import web_search

INTERVIEW_PREP_PROMPT = """You are an interview preparation assistant. Based on the search 
results below about a company's interview process, generate helpful, 
structured interview preparation guidance.

Search results about the interview PROCESS:
{process_results}

Search results about common interview QUESTIONS:
{questions_results}

Search results about candidate EXPERIENCES:
{experience_results}

Based ONLY on the information in these search results (do not invent 
information not supported by the results), extract and organize:

- probable_questions: a list of specific interview questions likely to 
  be asked, based on the search results
- interview_process: a list of steps/stages describing the typical 
  interview process, based on the search results
- interview_experiences: a list of tips or insights shared by people 
  who have interviewed there, based on the search results

If the search results don't contain enough information for a field, 
return a shorter list rather than inventing content.

Respond with ONLY valid JSON in exactly this format, and nothing else 
- no explanations, no markdown code blocks:

{{
  "probable_questions": ["question1", "question2"],
  "interview_process": ["step1", "step2"],
  "interview_experiences": ["tip1", "tip2"]
}}
"""


def get_interview_prep(company_name: str, job_role: str) -> dict:
    # Step 1: Run 3 targeted searches
    process_snippets = web_search(f"{company_name} {job_role} interview process")
    questions_snippets = web_search(f"{company_name} {job_role} interview questions")
    experience_snippets = web_search(f"{company_name} {job_role} interview experience")

    process_text = "\n\n".join(process_snippets)
    questions_text = "\n\n".join(questions_snippets)
    experience_text = "\n\n".join(experience_snippets)

    # Step 2: Feed combined results to the LLM to structure
    llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.2)

    prompt = INTERVIEW_PREP_PROMPT.format(
        process_results=process_text,
        questions_results=questions_text,
        experience_results=experience_text,
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
        raise ValueError(f"Interview Prep Agent returned invalid JSON: {e}")

    return parsed_result


if __name__ == "__main__":
    result = get_interview_prep(company_name="Google", job_role="Software Engineer")
    print(json.dumps(result, indent=2))