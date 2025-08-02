# Smart Resume Screener using NLP

import os
import glob
import pandas as pd
import numpy as np
import docx2txt
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

# ----------- CONFIG -----------
RESUME_FOLDER = "resumes/"
JOB_DESC_FILE = "job_description.txt"
TOP_K = 5

# ----------- UTILITIES -----------
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def get_all_resumes():
    resumes = []
    for file in glob.glob(os.path.join(RESUME_FOLDER, '*')):
        filename = os.path.basename(file)
        if file.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        elif file.endswith(".docx"):
            text = extract_text_from_docx(file)
        elif file.endswith(".txt"):
            with open(file, "r") as f:
                text = f.read()
        else:
            continue
        resumes.append((filename, text))
    return resumes


# ----------- MAIN LOGIC -----------

def get_keywords(text):
    doc = nlp(text.lower())
    keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(keywords)

def rank_resumes(job_desc, resumes):
    texts = [get_keywords(job_desc)] + [get_keywords(r[1]) for r in resumes]
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(texts)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    scored = [(resumes[i][0], round(float(cosine_sim[i]), 3)) for i in range(len(resumes))]
    return sorted(scored, key=lambda x: x[1], reverse=True)

if __name__ == '__main__':
    print("\n📂 Loading resumes from 'resumes/' folder...")
    resumes = get_all_resumes()
    if not resumes:
        print("No resumes found! Please add PDFs or DOCX files to the 'resumes/' folder.")
        exit()

    print("📝 Reading job description...")
    with open(JOB_DESC_FILE, 'r') as f:
        job_desc = f.read()

    print("⚙️  Ranking resumes...")
    ranked = rank_resumes(job_desc, resumes)

    print(f"\n📊 Top {TOP_K} Ranked Resumes:\n")
    for i, (name, score) in enumerate(ranked[:TOP_K], 1):
        print(f"{i}. {name} - Score: {score}")
