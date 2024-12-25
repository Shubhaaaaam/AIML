# 1. Importing the Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 2. Loading the Dataset
dataset = pd.read_csv("decision.csv")

# 3. Display the Number of Columns and Rows
print(f"Dataset shape: {dataset.shape}")

# 4. Statistical Data for Each Column
print(dataset.describe(include="all"))

# 5. Display Null Values in Each Column
print(dataset.isnull().sum())

# 6. Replace Null Values
imputer = SimpleImputer(strategy="mean")
numerical_columns = ["Price", "Number_of_Reviews", "Average_Review_Score", "Product_Age"]
dataset[numerical_columns] = imputer.fit_transform(dataset[numerical_columns])

# Replace null values in Brand_Popularity with the most frequent value
imputer = SimpleImputer(strategy="most_frequent")
dataset["Brand_Popularity"] = imputer.fit_transform(dataset[["Brand_Popularity"]]).ravel()


# 7. Encoding
label_encoder = LabelEncoder()
dataset["Category"] = label_encoder.fit_transform(dataset["Category"])
dataset = pd.get_dummies(dataset, columns=["Brand_Popularity"], drop_first=True)

# 8. No Scaling Needed for Decision Trees

# 9. Importing the Model
model = DecisionTreeClassifier()

# 10. Train-Test Split
X = dataset.drop("Category", axis=1)
y = dataset["Category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 11. Training the Model and Prediction
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 12. Performance Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))

# 13. Analysis of Results
if model.score(X_test, y_test) > 0.8:
    print("The decision tree performed well.")
else:
    print("Consider pruning the tree or collecting more data.")
