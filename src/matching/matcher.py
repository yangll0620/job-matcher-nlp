from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from typing import Dict

class JobMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.kw_model = KeyBERT(model_name)


    def match(self, resume_txt: str, jd_txt: str, top_n: int = 5) -> Dict[str, any]:
        """
        Match a resume to a job description and return similarity score and keyword analysis.

        Args:
            resume_text (str): The resume content.
            jd_text (str): The job description content.
            top_n (int): Number of keywords to extract.

        Returns:
            Dict[str, Any]: Dictionary with similarity score, matched keywords, and all extracted keywords.
        """
        # calculate similarity scores 
        resume_embedding = self.model.encode(resume_txt, convert_to_tensor=True)
        jd_embedding = self.model.encode(jd_txt, convert_to_tensor=True)
        score = util.cos_sim(resume_embedding, jd_embedding)

        # Extract keywords
        resume_keywords = self.kw_model.extract_keywords(resume_txt, top_n=top_n)
        jd_keywords = self.kw_model.extract_keywords(jd_txt, top_n=top_n)

        # Matched keywords
        jd_keywords_set = set([k for k, _ in jd_keywords])
        resume_keywords_set = set([k for k, _ in resume_keywords])
        matched_keywords = list(jd_keywords_set & resume_keywords_set)


        return {
            'similarity_scores': score.cpu().numpy(),
            'resume_keywords': resume_keywords_set,
            'jd_keywords': jd_keywords_set,
            'matched_keywords': matched_keywords
        }

if __name__ == "__main__":
    # Example usage
    resume_text = "Experienced software engineer with expertise in Python and machine learning."
    jd_text = "Looking for a software engineer with strong Python skills and experience in machine learning."

    matcher = JobMatcher()
    result = matcher.match(resume_txt=resume_text, jd_txt=jd_text, top_n=5)

    print("Similarity Score:\t", result['similarity_scores'])
    print("Resume Keywords:\t", result['resume_keywords'])
    print("Job Description Keywords:\t", result['jd_keywords'])
    print("Matched Keywords:\t", result['matched_keywords'])
