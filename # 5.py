import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
dataset = pd.read_csv("svm.csv")
print(f"Dataset shape: {dataset.shape}")
print(dataset.describe(include="all"))
print(dataset.isnull().sum())
imputer = SimpleImputer(strategy="mean")
numerical_columns = ["Age", "Annual_Income", "Credit_Score", "Loan_Amount", "Number_of_Dependents"]
dataset[numerical_columns] = imputer.fit_transform(dataset[numerical_columns])
label_encoder = LabelEncoder()
dataset["Will_Opt_Credit_Card"] = label_encoder.fit_transform(dataset["Will_Opt_Credit_Card"])
scaler = StandardScaler()
dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])
model = SVC()
X = dataset.drop("Will_Opt_Credit_Card", axis=1)
y = dataset["Will_Opt_Credit_Card"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
if accuracy_score(y_test, y_pred) > 0.8:
    print("The model performed well with high accuracy.")
else:
    print("Consider tuning hyperparameters or balancing the dataset.")
