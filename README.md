Smart resume screener- A smart NLP-based web app that ranks resumes against a job description using semantic similarity.
Built with React (frontend) and FastAPI (backend), this tool leverages SBERT to automate candidate shortlisting.


Features
- Parse resumes in PDF, DOCX, and TXT formats
- Compute semantic similarity using Sentence-BERT
- FastAPI backend for processing and ranking
- Responsive React + TailwindCSS frontend
- Instant ranking results with match scores



Setup

1. Clone the repo:
   git clone https://github.com/arjun-sagar/smart-resume-screener.git
   cd smart-resume-screener

2. Set up backend:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   uvicorn backend_api:app --reload

3. Run frontend:
   Open index.html in a browser.

   Make sure the backend is running at http://127.0.0.1:8000.



How It Works
1. Paste a Job Description.
2. Upload resumes (PDF/DOCX/TXT).
3. The frontend extracts text using pdf.js or mammoth.js.
4. Sends JD and resumes to FastAPI.
5. Backend encodes text with Sentence-BERT and computes cosine similarity.
6. Results are returned and displayed as ranked scores.


Tech Stack
Frontend: React, TailwindCSS, pdf.js, mammoth.js  
Backend: FastAPI, Python, SentenceTransformers  
