# 1. Importing the Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 2. Loading the Dataset
dataset = pd.read_csv("svm.csv")

# 3. Display the Number of Columns and Rows
print(f"Dataset shape: {dataset.shape}")

# 4. Statistical Data for Each Column
print(dataset.describe(include="all"))

# 5. Display Null Values in Each Column
print(dataset.isnull().sum())

# 6. Replace Null Values
imputer = SimpleImputer(strategy="mean")
numerical_columns = ["Age", "Annual_Income", "Credit_Score", "Loan_Amount", "Number_of_Dependents"]
dataset[numerical_columns] = imputer.fit_transform(dataset[numerical_columns])

# 7. Encoding
label_encoder = LabelEncoder()
dataset["Will_Opt_Credit_Card"] = label_encoder.fit_transform(dataset["Will_Opt_Credit_Card"])

# 8. Scaling
scaler = StandardScaler()
dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])

# 9. Importing the Model
model = SVC()

# 10. Train-Test Split
X = dataset.drop("Will_Opt_Credit_Card", axis=1)
y = dataset["Will_Opt_Credit_Card"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 11. Training the Model and Prediction
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 12. Performance Evaluation
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 13. Analysis of Results
if accuracy_score(y_test, y_pred) > 0.8:
    print("The model performed well with high accuracy.")
else:
    print("Consider tuning hyperparameters or balancing the dataset.")
