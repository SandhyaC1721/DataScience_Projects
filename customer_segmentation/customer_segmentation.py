# Customer Segmentation using K-Means

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Step 1: Load dataset
data = pd.read_csv("Mall_Customers.csv")

print("First 5 rows of dataset:")
print(data.head())

# Step 2: Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 3: Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# Step 4: Add cluster labels to dataset
data['Cluster'] = kmeans.labels_

print("\nClustered Data:")
print(data.head())

# Step 5: Visualize clusters
plt.figure()
plt.scatter(
    data['Annual Income (k$)'],
    data['Spending Score (1-100)'],
    c=data['Cluster']
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.show()