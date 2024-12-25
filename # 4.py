
# Step 1: Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 2: Loading the dataset using pandas
file_path = "Electronics.csv"  # Update the path if necessary
df = pd.read_csv(file_path)

# Step 3: Display the number of columns and rows in the dataset
print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

# Step 4: Knowing the statistical data for each column
print(df.describe(include='all'))

# Step 5: Display how many null values each column has
print("Null values per column:")
print(df.isnull().sum())

# Step 6: Methods to replace null values
# Fill numerical columns with their mean
numerical_cols = ["Age", "Annual_Income", "Monthly_Spend"]
for col in numerical_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill categorical columns with the mode
categorical_cols = ["Gender", "Electronics_Purchased"]
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Step 7: Encoding using one hot encoder or label encoder
# Label Encoding for Gender
label_encoder = LabelEncoder()
df["Gender"] = label_encoder.fit_transform(df["Gender"])  # Male: 1, Female: 0

# One-Hot Encoding for Electronics_Purchased
df = pd.get_dummies(df, columns=["Electronics_Purchased"], drop_first=True)

# Step 8: Scaling, normalization of the data
scaler = StandardScaler()  # Standard scaling
scaled_cols = ["Age", "Annual_Income", "Monthly_Spend"]
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# Step 9: Importing the model
model = LinearRegression()

# Step 10: Defining train test split
X = df.drop(columns=["Customer_ID", "Sales"])  # Features
y = df["Sales"]  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 11: Defining X and Y variables
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# Step 12: Prediction for given values
model.fit(X_train, y_train)  # Train the model
y_pred = model.predict(X_test)  # Predict on the test set

# Step 13: Performance evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")

# Step 14: Analysis of the results
print("\nAnalysis:")
print(f"The model explains {r2 * 100:.2f}% of the variance in the target variable.")
print("Lower MSE and MAE indicate better model performance. Evaluate residuals for more insight.")
