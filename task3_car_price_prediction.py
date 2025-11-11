import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

data = pd.read_csv("car data.csv")
print("✅ Dataset loaded successfully!\n")
print(data.head())


print("\nDataset Info:")
print(data.info())

print("\nMissing Values:\n", data.isnull().sum())


print("\nSummary Statistics:\n", data.describe())


plt.figure(figsize=(8, 5))
sns.histplot(data["Selling_Price"], kde=True, color="teal")
plt.title("Distribution of Car Selling Prices")
plt.xlabel("Selling Price (in lakhs)")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x="Fuel_Type", y="Selling_Price", data=data, palette="Set2")
plt.title("Fuel Type vs Selling Price")
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(x="Year", y="Selling_Price", data=data, color="orange")
plt.title("Car Price vs Manufacturing Year")
plt.show()

data_encoded = pd.get_dummies(data, drop_first=True)


X = data_encoded.drop("Selling_Price", axis=1)
y = data_encoded["Selling_Price"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n✅ Data split into training and testing sets successfully!")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

print("\n--- Model Performance ---")
print("Linear Regression R² Score:", round(r2_score(y_test, lr_pred), 3))
print("Random Forest R² Score:", round(r2_score(y_test, rf_pred), 3))
print("Mean Absolute Error (RF):", round(mean_absolute_error(y_test, rf_pred), 3))
print("Root Mean Squared Error (RF):", round(np.sqrt(mean_squared_error(y_test, rf_pred)), 3))


plt.figure(figsize=(7, 5))
sns.scatterplot(x=y_test, y=rf_pred, color="royalblue")
plt.title("Actual vs Predicted Car Prices (Random Forest)")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.show()


print("""
✅ Analysis Complete!


