import joblib
import numpy as np

# Modeli yükle
model = joblib.load("xgboost_wine_model.pkl")
print("Model başarıyla yüklendi.")

# Yeni veri üzerinde tahmin yap
sample_input = np.array([[6.5,0.3,0.27,4.0,0.038,37.0,97.0,0.99026,3.2,0.6,12.6]])
prediction = model.predict(sample_input)
print(f"Tahmin Edilen Kalite: {prediction[0]:.2f}")
