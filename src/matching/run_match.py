import os
import pandas as pd

from src.matching.matcher import JobMatcher


def read_text_file(file_path: str) -> str:
    """
    Read the content of a text file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: Content of the file.
    """
    with open(file_path, 'r', encoding='utf-8', errors="ignore") as file:
        return file.read()

def read_text_from_folder(folder_path: str) -> list:
    """
    Read all text files in a folder and return their contents.

    Args:
        folder_path (str): Path to the folder containing text files.

    Returns:
        list: List of strings, each representing the content of a text file.
    """
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            texts[filename] = read_text_file(file_path)
    return texts


def match_resume_to_jd(resume_path: str, jd_path: str, top_n: int = 5):
    resume_text = read_text_file(resume_path)
    jd_text = read_text_file(jd_path)

    matcher = JobMatcher()

    result = matcher.match(resume_txt=resume_text, jd_txt=jd_text, top_n=top_n)

    return result

def match_resume_to_multiple_jds(resume_path: str, jd_dir: str, top_n: int = 5) -> dict[str, dict]:
    """
    Matches a resume to multiple job descriptions (JDs) and returns the top N matches for each JD, sorted by similarity score.

    Args:
        resume_path (str): Path to the resume text file.
        jd_dir (str): Directory containing job description text files.
        top_n (int, optional): Number of top matches to return for each JD. Defaults to 5.

    Returns:
        dict[str, dict]: A dictionary where keys are JD filenames and values are match results (dict),
              sorted in descending order by similarity score.
    """
    resume_text = read_text_file(resume_path)
    jd_texts = read_text_from_folder(jd_dir)

    matcher = JobMatcher()

    results = {}
    for jd_filename, jd_text in jd_texts.items():
        result = matcher.match(resume_txt=resume_text, jd_txt=jd_text, top_n=top_n)
        results[jd_filename] = result

    # Sorted results by similarity score
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['similarity_scores'], reverse=True))

    return sorted_results

if __name__ == "__main__":
    resume_path = 'data/resumes/resume_sde_backend.txt'
    jd_dir = 'data/jobs'
    top_n = 5

    results = match_resume_to_multiple_jds(resume_path, jd_dir, top_n)

    print(f"Resume: {resume_path}")
    for jd, matched in results.items():
        print(f" > Match with {jd:<40} {matched['similarity_scores']:<10.4f} mathed keywords: {matched['matched_keywords']}, resume_keywords: {matched['resume_keywords']}, jd_keywords: {matched['jd_keywords']}")
