# ==============================
# TITANIC END-TO-END ML PROJECT
# ==============================

# 1️⃣ Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 2️⃣ Load dataset
data = pd.read_csv("titanic.csv")

print("First 5 rows of dataset:")
print(data.head())


# 3️⃣ Basic Information
print("\nDataset Info:")
print(data.info())


# 4️⃣ Data Cleaning

# Fill missing Age with median age
data['Age'].fillna(data['Age'].median(), inplace=True)

# Fill missing Embarked with most common value
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop Cabin column (too many missing values)
data.drop('Cabin', axis=1, inplace=True)


# 5️⃣ Convert categorical columns to numbers

# Convert Sex (male=0, female=1)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked to numbers
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)


# 6️⃣ Select Features (Input) and Target (Output)

X = data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = data['Survived']


# 7️⃣ Split Data into Train and Test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 8️⃣ Train Logistic Regression Model

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# 9️⃣ Make Predictions

predictions = model.predict(X_test)


# 🔟 Evaluate Model

accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# 1️⃣1️⃣ Visualization (Optional but Good for Project)

plt.figure()
sns.countplot(x='Survived', data=data)
plt.title("Survival Count")
plt.show()