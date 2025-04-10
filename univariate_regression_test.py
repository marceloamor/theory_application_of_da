import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from icecream import ic
# generate random non linear data
x = np.linspace(0,10,100)
y = x**2 + np.random.normal(0,1,100)

# plot the data
plt.scatter(x,y)
plt.savefig('non_linear_data.png')

# fit a linear regression model
model = LinearRegression()
x = x.reshape(-1, 1)  # or np.array(x).reshape(-1, 1)
model.fit(x,y)

# polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
model_poly = LinearRegression()
model_poly.fit(x_poly,y)

poly3 = PolynomialFeatures(degree=3)
x_poly3 = poly3.fit_transform(x)
model_poly3 = LinearRegression()
model_poly3.fit(x_poly3,y)

# Create the plot with both linear and polynomial regression
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)  # Plot original data points
plt.plot(x, model.predict(x), color='red', label='Linear Regression')
plt.plot(x, model_poly.predict(x_poly), color='blue', label='Polynomial Regression')
plt.plot(x, model_poly3.predict(x_poly3), color='green', label='Polynomial Regression 3')
plt.xlabel('Distance')
plt.ylabel('Correlation')
plt.title('Linear vs Polynomial Regression: Distance vs Correlation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('linear_regression.png')

# Print regression statistics
ic(f"R² Score: {model.score(x, y):.3f}")
ic(f"Slope: {model.coef_[0]:.3f}")
ic(f"Intercept: {model.intercept_:.3f}")

# Print polynomial regression statistics 
ic(f"R² Score: {model_poly.score(x_poly, y):.3f}")
ic(f"Slope: {model_poly.coef_[0]:.3f}")
ic(f"Intercept: {model_poly.intercept_:.3f}")

# Print polynomial regression statistics 3
ic(f"R² Score: {model_poly3.score(x_poly3, y):.3f}")
ic(f"Slope: {model_poly3.coef_[0]:.3f}")
ic(f"Intercept: {model_poly3.intercept_:.3f}")


residuals = x - model.predict(x)

ic(residuals)

# plot residuals
plt.scatter(x, residuals)

plt.savefig('residuals.png')