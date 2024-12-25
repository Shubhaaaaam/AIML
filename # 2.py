import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
dataset = pd.read_csv("kmeans.csv")
print(f"Dataset shape: {dataset.shape}")
print(dataset.describe())
print(dataset.isnull().sum())
imputer = SimpleImputer(strategy="mean")
dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
scaler = StandardScaler()
dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)
model = KMeans(n_clusters=3, random_state=42)
model.fit(dataset)
cluster_labels = model.predict(dataset)
print("Silhouette Score:", silhouette_score(dataset, cluster_labels))
if silhouette_score(dataset, cluster_labels) > 0.5:
    print("The clusters are well-defined.")
else:
    print("Consider changing the number of clusters or scaling methods.")
