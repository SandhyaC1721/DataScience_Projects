# Import libraries
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load English model
nlp = spacy.load("en_core_web_sm")

# Read resume file
with open("resume.txt", "r") as file:
    resume_text = file.read()

# Read job description file
with open("job_description.txt", "r") as file:
    job_text = file.read()

# Convert text into numbers using TF-IDF
vectorizer = TfidfVectorizer()

# Fit and transform both texts
tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])

# Calculate similarity
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

# Print result
print("Resume Match Percentage:", round(similarity[0][0] * 100, 2), "%")