# Step 1: Import libraries
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Step 3: Load dataset
data = pd.read_csv("IMDB Dataset.csv")

# Step 4: Clean and convert sentiment labels
data['sentiment'] = data['sentiment'].str.strip().str.lower()
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Remove any missing values (important)
data = data.dropna(subset=['sentiment'])

# Step 5: Split into input and output
X = data['review']
y = data['sentiment']

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Convert text into numbers
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# Step 8: Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vector, y_train)

# Step 9: Make predictions
y_pred = model.predict(X_test_vector)

# Step 10: Print results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Step 11: Test custom review
sample_review = ["This movie was absolutely fantastic and amazing!"]
sample_vector = vectorizer.transform(sample_review)
prediction = model.predict(sample_vector)

if prediction[0] == 1:
    print("\nCustom Review Prediction: Positive")
else:
    print("\nCustom Review Prediction: Negative")
