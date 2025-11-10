from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


class ResumeData(BaseModel):
    filename: str
    text: str

class RankRequest(BaseModel):
    jd_text: str
    resumes: List[ResumeData]

print("Loading SBERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("SBERT Model loaded successfully.")


app = FastAPI(
    title="Semantic Resume Ranker API",
    description="An API to rank resumes against a job description using SBERT."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/rank")
def rank_resumes(request: RankRequest):
    """
    Receives a job description and a list of resumes,
    returns a ranked list of resumes.
    """
    print(f"Received ranking request for {len(request.resumes)} resumes.")
    
    
    jd_text = request.jd_text
    resume_texts = [resume.text for resume in request.resumes]
    resume_filenames = [resume.filename for resume in request.resumes]
    
    if not jd_text or not resume_texts:
        return {"error": "No job description or resumes provided."}

    try:
        
        print("Creating embeddings...")
        jd_embedding = model.encode(jd_text)
        resume_embeddings = model.encode(resume_texts)
        
       
        print("Calculating similarity...")
        jd_embedding_2d = jd_embedding.reshape(1, -1)
        similarities = cosine_similarity(
            jd_embedding_2d,
            resume_embeddings
        )
        scores = similarities.flatten()

        
        print("Formatting results...")
        ranked_results = []
        for i in range(len(resume_filenames)):
            ranked_results.append({
                "filename": resume_filenames[i],
                "score": float(scores[i]) # Convert numpy.float32 to standard float
            })

        
        ranked_results.sort(key=lambda x: x["score"], reverse=True)

        return {"ranked_resumes": ranked_results}
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

