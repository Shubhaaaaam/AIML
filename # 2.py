# 1. Importing the Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 2. Loading the Dataset
dataset = pd.read_csv("kmeans.csv")

# 3. Display the Number of Columns and Rows
print(f"Dataset shape: {dataset.shape}")

# 4. Statistical Data for Each Column
print(dataset.describe())

# 5. Display Null Values in Each Column
print(dataset.isnull().sum())

# 6. Replace Null Values
imputer = SimpleImputer(strategy="mean")
dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)

# 7. No Encoding Needed for Numerical Features

# 8. Scaling
scaler = StandardScaler()
dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)

# 9. Importing the Model
model = KMeans(n_clusters=3, random_state=42)

# 10. No Train-Test Split for Clustering

# 11. Fit the Model and Predict
model.fit(dataset)
cluster_labels = model.predict(dataset)

# 12. Performance Evaluation
print("Silhouette Score:", silhouette_score(dataset, cluster_labels))

# 13. Analysis of Results
if silhouette_score(dataset, cluster_labels) > 0.5:
    print("The clusters are well-defined.")
else:
    print("Consider changing the number of clusters or scaling methods.")
