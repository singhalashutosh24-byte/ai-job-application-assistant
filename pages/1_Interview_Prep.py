import streamlit as st

st.title("Interview Preparation")

if "final_state" not in st.session_state:
    st.warning("No results yet. Please go to the main page and generate results first.")
else:
    result = st.session_state["final_state"]

    st.write(f"Prep guidance for **{result['job_role']}** at **{result['company_name']}**")

    tab1, tab2, tab3 = st.tabs(["Probable Questions", "Interview Process", "Tips & Experiences"])

    with tab1:
        st.subheader("Likely Interview Questions")
        for question in result["probable_questions"]:
            st.write(f"- {question}")

    with tab2:
        st.subheader("Interview Process")
        for step in result["interview_process"]:
            st.write(f"- {step}")

    with tab3:
        st.subheader("Tips from Past Candidates")
        for tip in result["interview_experiences"]:
            st.write(f"- {tip}")