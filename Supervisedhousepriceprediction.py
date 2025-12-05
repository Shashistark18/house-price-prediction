import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Creating a small dataset of house prices
data = {'Square_Feet': [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000],
        'Price': [150000, 180000, 200000, 250000, 280000, 300000, 320000, 350000, 380000, 400000]}

df = pd.DataFrame(data)

# Display dataset
print(df)

# Selecting Features (X) and Target (y)
X = df[['Square_Feet']]
y = df['Price']

# Splitting Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on Test Data
y_pred = model.predict(X_test)

# Show Predictions
results = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
print(results)

# Calculate Error (Lower is better)
error = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: ${error:.2f}")

# Plot the Regression Line
plt.scatter(X, y, color='blue', label='Actual Prices')
plt.plot(X, model.predict(X), color='red', label='Predicted Line')
plt.xlabel("Square Feet")
plt.ylabel("Price ($)")
plt.title("House Price Prediction")
plt.legend()
plt.show()

# Example: Predict price for a house of 2100 sq ft
new_house = [[2100]]
predicted_price = model.predict(new_house)
print(f"Predicted price for a 2100 sq ft house: ${predicted_price[0]:.2f}")





#pip install pandas numpy scikit-learn matplotlib
