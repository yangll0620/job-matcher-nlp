import os
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

def build_prompt(resume_text: str, jd_text: str) -> str:
    prompt = f"""
            Your are a professinal resume writer.

            Here is a resume:
            ---
                {resume_text}
            ---

            Here is a job description:
            ---
                {jd_text}
            ---
            Rewrite the work experience section of the resume to better match the job description. 
            Keep it truthful, highlight relevant skills, and write in concise, achievement-based bullet points.
            """
    return prompt

def generate_tailored_resume(resume_text:str, jd_text:str, model_name:str="gpt-3.5-turbo") -> str:
    
    # Get the prompt
    prompt = build_prompt(resume_text, jd_text)


    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a professional career assistant"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    # Example usage
    resume_text = "Experienced software engineer with a strong background in Python and machine learning."
    jd_text = "Looking for a software engineer with expertise in Python, machine learning, and data analysis."
    
    tailored_resume = generate_tailored_resume(resume_text, jd_text)
    print("Tailored Resume:\n", tailored_resume)

