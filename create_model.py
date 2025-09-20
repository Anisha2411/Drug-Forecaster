import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

# Create and train a simple drug rating prediction model
model = LinearRegression()

# Sample training data
# Format: [number_of_side_effects, pregnancy_category_encoded]
X = np.array([
    [3, 1],  # Few side effects, pregnancy category A
    [5, 2],  # Some side effects, pregnancy category B
    [8, 3],  # Moderate side effects, pregnancy category C
    [12, 4], # Many side effects, pregnancy category D
    [15, 5]  # Many side effects, pregnancy category X
])

# Corresponding ratings (on a scale of 1-10, where 10 is safest)
y = np.array([8.5, 7.2, 5.8, 3.2, 1.5])

# Train the model
model.fit(X, y)

# Save the model to a file
joblib.dump(model, "rating_model.pkl")
print("Model created and saved as rating_model.pkl")
print("Model coefficients:", model.coef_)
print("Model intercept:", model.intercept_)