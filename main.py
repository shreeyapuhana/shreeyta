
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
#load the data set
df = pd.read_csv('/content/housing.csv')
print(df.head())

# Feature and target selection
X = df[["median_income"]]
y = df["median_house_value"]

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict using the model
y_pred = model.predict(X)

# Plotting the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.3, label="Actual data")
plt.plot(X, y_pred, color='red', label="Regression line")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Linear Regression: Predicting House Value from Median Income")
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Print evaluation metrics
print("Model Performance Metrics:")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared (R²): {r2:.4f}")