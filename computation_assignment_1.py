import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

# Q2:
x = np.array([1.3, 0.7, 0.0, -0.7, -1.3,-1.6, -2.0, -1.6, -1.3,-0.7, 0.0, 0.7, 1.3, 1.6, 2.0, 1.6])

y = np.array([2.7, 2.6, 2.5, 1.5, 0.7, -0.3, -1.5, -2.1, -2.7, -2.6, -2.5, -1.5, -0.7, 0.3, 1.5, 2.1])

#calculate empirical mean and standard deviation of x and y
mean_x = np.mean(x)
mean_y = np.mean(y)
std_x = np.std(x)
std_y = np.std(y)

# Calculate means manually
mean_x = sum(x) / len(x)
mean_y = sum(y) / len(y)

# Calculate standard deviations manually
# Step 1: Calculate sum of squared differences from mean
squared_diff_x = sum((xi - mean_x) ** 2 for xi in x)
squared_diff_y = sum((yi - mean_y) ** 2 for yi in y)

# Step 2: Calculate standard deviation (using n-1 for sample standard deviation)
std_x = np.sqrt(squared_diff_x / (len(x) - 1))
std_y = np.sqrt(squared_diff_y / (len(y) - 1))

# Print results
print(f"X array:")
print(f"Mean: {mean_x:.4f}")
print(f"Standard Deviation: {std_x:.4f}")

print(f"\nY array:")
print(f"Mean: {mean_y:.4f}")
print(f"Standard Deviation: {std_y:.4f}")

# Verify with numpy
print("\nVerification with numpy:")
print(f"X: mean={np.mean(x):.4f}, std={np.std(x):.4f}")
print(f"Y: mean={np.mean(y):.4f}, std={np.std(y):.4f}")

# Calculate the covariance matrix
cov_matrix = np.cov(x, y)
ic(cov_matrix)

# manually run a PCA analysis
# standardize the data

# Standardize x and y arrays (z-score normalization)
x_standardized = (x - mean_x) / std_x
y_standardized = (y - mean_y) / std_y

# Verify the standardization worked correctly
# Standardized data should have mean ≈ 0 and std ≈ 1
print("Verification of standardized data:")
print("\nX standardized:")
print(f"Mean: {np.mean(x_standardized):.10f}")  # Should be very close to 0
print(f"Std: {np.std(x_standardized):.10f}")    # Should be very close to 1

print("\nY standardized:")
print(f"Mean: {np.mean(y_standardized):.10f}")  # Should be very close to 0
print(f"Std: {np.std(y_standardized):.10f}")    # Should be very close to 1

# Store standardized data in a 2D array for PCA
standardized_data = np.vstack((x_standardized, y_standardized)).T
print("\nShape of standardized data:", standardized_data.shape)
ic(standardized_data)

# calculate the covariance matrix
cov_matrix_pca = np.cov(x_standardized, y_standardized)
ic(cov_matrix_pca)

# calculate the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix_pca)
ic(eigenvalues, eigenvectors)

# plot the original data and the PCA data
plt.scatter(x, y)
plt.scatter(x_standardized, y_standardized)
plt.savefig('pca_data.png')

#####################################

# ## Q3:
# x = np.array([27.5, 20.8, 33.2, 18.3, 29.9, 24.1])
# y = np.array([12.5, 9.6, 14.7, 8.4, 13.9, 11.1])

# # Calculate means
# x_mean = np.mean(x)
# ic(x_mean)
# y_mean = np.mean(y)
# ic(y_mean)

# # Calculate regression coefficients
# # slope (m) = Σ((x - x_mean)(y - y_mean)) / Σ((x - x_mean)²)
# numerator = np.sum((x - x_mean) * (y - y_mean))
# ic(numerator)
# denominator = np.sum((x - x_mean) ** 2)
# ic(denominator)
# m = numerator / denominator
# ic(m)

# # Calculate y-intercept (b) using y = mx + b
# b = y_mean - m * x_mean

# # Print the equation
# print(f"Linear Regression Equation: y = {m:.4f}x + {b:.4f}")

# # standard error of regression coefficients
# # Calculate standard error of regression coefficients
# # Calculate predicted y values
# y_pred = m * x + b
# ic(y_pred)
# # First calculate residuals
# residuals = y - y_pred

# # Calculate variance of residuals (MSE)
# n = len(x)
# mse = np.sum(residuals**2) / (n-2)  # n-2 degrees of freedom for simple linear regression
# ic(mse)

# # Standard error of slope calculation:
# # SE(m) = sqrt(MSE / Σ(x - x̄)²)
# # where:
# # - MSE = Σ(y - ŷ)² / (n-2)  [Mean Square Error]
# # - ŷ = mx + b  [predicted y values]
# # - n-2 degrees of freedom (losing one for slope and one for intercept)
# #
# # Working out:
# # 1. Calculate residuals: e = y - ŷ
# # 2. Square residuals: e²
# # 3. Sum squared residuals: Σe²
# # 4. Divide by (n-2) to get MSE
# # 5. Divide MSE by Σ(x - x̄)² 
# # 6. Take square root of result

# # Standard error of slope (m)
# se_m = np.sqrt(mse / np.sum((x - x_mean)**2))
# ic(np.sum((x - x_mean)**2))
# ic(se_m)

# # Standard error of intercept calculation:
# # SE(b) = sqrt(MSE * (1/n + x̄²/Σ(x - x̄)²))
# # where:
# # - MSE = Σ(y - ŷ)² / (n-2)  [Mean Square Error]
# # - x̄ is the mean of x values
# # - n is the sample size
# # - Σ(x - x̄)² is the sum of squared deviations of x from its mean
# #
# # Working out:
# # 1. Calculate MSE as before
# # 2. Calculate 1/n
# # 3. Calculate x̄²/Σ(x - x̄)²
# # 4. Add these terms inside the square root
# # 5. Multiply by MSE
# # 6. Take square root of result

# # Standard error of intercept (b)
# se_b = np.sqrt(mse * (1/n + x_mean**2/np.sum((x - x_mean)**2)))
# ic(se_b)

# print(f"Standard error of slope: {se_m:.4f}")
# print(f"Standard error of intercept: {se_b:.4f}")

# # sum  of squared errors for x
# sum_squared_errors_x = np.sum((x - x_mean)**2)
# ic(sum_squared_errors_x)




# # Calculate R-squared
# ss_total = np.sum((y - y_mean) ** 2)
# ss_residual = np.sum((y - y_pred) ** 2)
# r_squared = 1 - (ss_residual / ss_total)
# ic(ss_total)
# ic(ss_residual)
# ic(r_squared)
# print(f"R-squared: {r_squared:.4f}")

# # Create the plot
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, color='blue', label='Data points')
# plt.plot(x, y_pred, color='red', label='Regression line')
# plt.xlabel('X values')
# plt.ylabel('Y values')
# plt.title('Linear Regression Model')
# plt.legend()
# plt.grid(True)
# plt.savefig('linear_regression.png')
# plt.close()


