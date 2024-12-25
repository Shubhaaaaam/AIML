import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
dataset = pd.read_csv("knn.csv")
print(f"Dataset shape: {dataset.shape}")
print(dataset.describe(include="all"))
print(dataset.isnull().sum())
imputer = SimpleImputer(strategy="mean")
dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
scaler = StandardScaler()
scaled_columns = ["Square_Footage", "Number_of_Bedrooms", "Number_of_Bathrooms", "Distance_to_City_Center", "Age_of_the_House"]
dataset[scaled_columns] = scaler.fit_transform(dataset[scaled_columns])
model = KNeighborsRegressor(n_neighbors=5)
X = dataset.drop("Selling_Price", axis=1)
y = dataset["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
if mean_squared_error(y_test, y_pred) < 50000:
    print("The model's predictions are accurate.")
else:
    print("Consider tuning hyperparameters or using more features.")
