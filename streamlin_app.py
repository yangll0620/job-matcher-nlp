import streamlit as st
from sentence_transformers import SentenceTransformer, util

from src.matching.matcher import JobMatcher

TOP_N = 10  # Number of keywords to extract

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

else:
    st.info("👈 Please upload a resume and paste a job description to see the match!")