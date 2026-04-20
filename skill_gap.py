import re
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")


def extract_from_pdf(path):
    text = ""
    reader = PyPDF2.PdfReader(path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def extract_dynamic_skills(text):
    """
    Extract potential skills dynamically using NLP.
    """
    doc = nlp(text.lower())
    skills = set()

    # Extract noun phrases
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if len(phrase) > 2 and len(phrase.split()) <= 3:
            skills.add(phrase)

    # Extract proper nouns / technical terms
    for token in doc:
        if token.pos_ in ["PROPN", "NOUN"]:
            if len(token.text) > 2:
                skills.add(token.text)

    return list(skills)


def extract_skill_from_jd(jd_text):
    return extract_dynamic_skills(jd_text)


def extract_skill_from_resume(resume_text, jd_skills):
    resume_text = resume_text.lower()
    return [skill for skill in jd_skills if re.search(rf"\b{re.escape(skill)}\b", resume_text)]


def skill_similarity(jd_text, resume_text):

    jd_skills = extract_skill_from_jd(jd_text)
    resume_skills = extract_skill_from_resume(resume_text, jd_skills)

    jd_str = " ".join(jd_skills)
    resume_str = " ".join(resume_skills)

    if not jd_str or not resume_str:
        return 0.0

    tfidf = TfidfVectorizer()
    vector = tfidf.fit_transform([jd_str, resume_str])

    score = cosine_similarity(vector[0:1], vector[1:2])[0][0]

    return round(score * 100, 2)


def skill_gap_analysis(jd_text, resume_text):

    jd_skills = extract_skill_from_jd(jd_text)
    resume_skills = extract_skill_from_resume(resume_text, jd_skills)

    missing = list(set(jd_skills) - set(resume_skills))
    matched = list(set(jd_skills) & set(resume_skills))

    return {
        "matched": matched,
        "missing": missing
    }