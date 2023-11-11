import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# data set
data = {
    'Year': [2018, 2019, 2020, 2021, 2022],
    'AccidentRate': [30497, 21969, 23590,22687, 21185]
}

# data frame
df = pd.DataFrame(data)

# Extracting the datas from the current variable 
X = df['Year'].values
y = df['AccidentRate'].values

# Non-linear custom data set
def custom_curve(x, a, b, c):
    return a * x**2 + b * x + c

# Fit the custom curve to the data using curve_fit
params, covariance = curve_fit(custom_curve, X, y)

# Extract the fitted parameters
a, b, c = params

# Predict accident rates for the next 5 years
future_years = np.array([2023, 2024, 2025, 2026, 2027])
predicted_rates = custom_curve(future_years, a, b, c)

for year, rate in zip(future_years, predicted_rates):
    print(f"Predicted accident rate for {year}: {rate:.2f}")

# Create a regression plot, a histogram of original data, and a histogram of predicted values
plt.figure(figsize=(18, 6))

# Regression plot
plt.subplot(1, 3, 1)
X_fit = np.linspace(min(X), max(X), 100)
y_fit = custom_curve(X_fit, a, b, c)
plt.scatter(X, y, label='Actual Data', color='blue')
plt.plot(X_fit, y_fit, color='red', label='Custom Curve Fitting')
plt.xlabel('Year')
plt.ylabel('Accident Rate')
plt.legend()

# Histogram of original data
plt.subplot(1, 3, 2)
plt.hist(y, bins=5, edgecolor='black', alpha=0.7, color='green')
plt.xlabel('Accident Rate (Original Data)')
plt.ylabel('Frequency')

# Histogram of predicted values
plt.subplot(1, 3, 3)
plt.hist(predicted_rates, bins=5, edgecolor='black', alpha=0.7, color='red')
plt.xlabel('Predicted Accident Rate')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
