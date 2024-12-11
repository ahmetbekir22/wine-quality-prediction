import joblib
import numpy as np

# Load model
model = joblib.load("xgboost_wine_model.pkl")
print("Model successfully loaded.")

# Make predictions
sample_input = np.array([[6.5,0.3,0.27,4.0,0.038,37.0,97.0,0.99026,3.2,0.6,12.6]])
prediction = model.predict(sample_input)
print(f"Estimated Quality: {prediction[0]:.2f}")
