import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
file_path = "Electronics.csv"
df = pd.read_csv(file_path)
print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
print(df.describe(include='all'))
print("Null values per column:")
print(df.isnull().sum())
numerical_cols = ["Age", "Annual_Income", "Monthly_Spend"]
for col in numerical_cols:
    df[col].fillna(df[col].mean(), inplace=True)
categorical_cols = ["Gender", "Electronics_Purchased"]
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
label_encoder = LabelEncoder()
df["Gender"] = label_encoder.fit_transform(df["Gender"])
df = pd.get_dummies(df, columns=["Electronics_Purchased"], drop_first=True)
scaler = StandardScaler()
scaled_cols = ["Age", "Annual_Income", "Monthly_Spend"]
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])
model = LinearRegression()
X = df.drop(columns=["Customer_ID", "Sales"])
y = df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")
print("\nAnalysis:")
print(f"The model explains {r2 * 100:.2f}% of the variance in the target variable.")
print("Lower MSE and MAE indicate better model performance. Evaluate residuals for more insight.")
