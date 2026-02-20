# STEP 1: Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# STEP 2: Load dataset (IMPORTANT: add sep=" ")
data = pd.read_csv("german_credit_data.csv", sep=" ", header=None)

# STEP 3: Separate features and target
X = data.iloc[:, :-1]   # all columns except last
y = data.iloc[:, -1]    # last column is target

# Convert target (1 = good, 2 = bad) to 0 and 1
y = y.replace({1: 0, 2: 1})

# STEP 4: Convert categorical values to numbers
le = LabelEncoder()

for col in X.columns:
    X[col] = le.fit_transform(X[col])

# STEP 5: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 6: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# STEP 7: Predict
y_pred = model.predict(X_test)

# STEP 8: Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

y_prob = model.predict_proba(X_test)[:, 1]
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))