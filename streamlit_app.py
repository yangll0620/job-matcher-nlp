import streamlit as st
from sentence_transformers import SentenceTransformer, util

from src.matching.matcher import JobMatcher
from src.inference.llm_inference import generate_tailored_resume

TOP_N = 10  # Number of keywords to extract
LLM_MODEL_NAME = "gpt-3.5-turbo"  # Default LLM model name

# Utility: clean and extract text from uploaded resume
def read_uploaded_file_as_text(file) -> str:
    """
    Reads the content of an uploaded file-like object (binary mode) 
    and returns it as a UTF-8 decoded string.

    Args:
        file: A file-like object (e.g., from Flask's request.files).

    Returns:
        str: The decoded text content of the uploaded file.
    """
    content = file.read().decode("utf-8", errors="ignore")
    return content

# Title
st.title("🔍 Job Matcher - Resume vs Job Description")

# Resume Upload
resume_file = st.file_uploader("📄 Upload your resume (.txt only)", type=["txt"])
jd_text = st.text_area("📝 Paste Job Description here", height=200)

# Action
matcher = JobMatcher()
if resume_file and jd_text:
    resume_text = read_uploaded_file_as_text(resume_file)

    if st.button("Match Resume with Job Description"):

        matched = matcher.match(resume_text, jd_text, top_n=TOP_N)
        score = matched["similarity_scores"]
        matched_keywords = matched["matched_keywords"]

        # Display results
        st.subheader("📊 Similarity Score")
        st.markdown(f"### Matching Score: **{score * 100:.2f}%**")
        st.markdown("---")

        st.subheader("🔑 Top Overlapping Keywords")
        if matched_keywords:
            st.markdown(", ".join(matched_keywords))
        else:
            st.markdown("_No significant overlap found.")
    
    if st.button("Generate Tailored Resume"):
        with st.spinner(f"Generating tailored using {LLM_MODEL_NAME} resume..."):
            tailored_resume = generate_tailored_resume(resume_text, jd_text, model_name=LLM_MODEL_NAME)
        
        st.subheader("🔍 Compare Resume: Before vs After")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📄 Original Resume")
            st.text_area("Original", value=resume_text, height=300)

        with col2:
            st.markdown("#### ✨ Tailored Resume")
            st.text_area("Tailored", value=tailored_resume, height=300)


        # Download Tailored Resume 
        st.download_button(
            label="Download Tailored Resume",
            data=tailored_resume,
            file_name="tailored_resume.txt",
            mime="text/plain"
        )

else:
    st.info("👈 Please upload a resume and paste a job description to see the match!")