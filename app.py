import streamlit as st
from graph import app as pipeline_app
from pypdf import PdfReader

st.title("AI Job Application Assistant")
st.write("Upload your resume, paste a job description, and get a tailored resume plus interview prep.")

resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste the job description here", height=200)
company_name = st.text_input("Company name")

generate_button = st.button("Generate")

if generate_button:
    if resume_file is None or not job_description.strip() or not company_name.strip():
        st.warning("Please fill in all fields before generating.")
    else:
        # Extract text from the uploaded PDF
        reader = PdfReader(resume_file)
        resume_text = ""
        for page in reader.pages:
            resume_text += page.extract_text()

        initial_state = {
            "resume_text": resume_text,
            "job_description": job_description,
            "company_name": company_name,
            "retry_count": 0,
        }

        with st.spinner("Running the pipeline... this may take a minute or two."):
            final_state = pipeline_app.invoke(initial_state)

        # Store results in session_state so other pages can access them
        st.session_state["final_state"] = final_state

        st.success(f"Done! ATS Score: {final_state['score']}/100 (Passed: {final_state['passed']})")

# Show resume results if we have them
if "final_state" in st.session_state:
    result = st.session_state["final_state"]

    st.header("Tailored Resume Bullets")
    for bullet in result["tailored_bullets"]:
        st.write(f"- {bullet}")

    st.header("ATS Score Details")
    st.metric("Score", f"{result['score']}/100")
    st.write(f"Attempts taken: {result['retry_count']}")
    if result["missing_keywords"]:
        st.write("Missing keywords:", ", ".join(result["missing_keywords"]))
    st.write("Improvement notes:", result["improvement_notes"])