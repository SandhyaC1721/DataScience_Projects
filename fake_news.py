# Step 1: Import libraries
import pandas as pd
import numpy as np
import nltk
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load dataset
data = pd.read_csv("Fake.csv")   # change file name if needed

# Step 3: Add label column (0 = Fake)
data["label"] = 0

# Step 4: Load real news dataset
real_data = pd.read_csv("True.csv")
real_data["label"] = 1   # 1 = Real

# Step 5: Combine both datasets
data = pd.concat([data, real_data], axis=0)

# Step 6: Use only text column
X = data["text"]
y = data["label"]

# Step 7: Convert text into numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# Step 8: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 9: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 10: Make predictions
y_pred = model.predict(X_test)

# Step 11: Check accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))