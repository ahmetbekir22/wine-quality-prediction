import pandas as pd
import numpy as np
import joblib

class WineQualityPredictor:
    def create_advanced_features(self, X):
        X_advanced = X.copy()
        X_advanced['alcohol_sugar_interaction'] = X_advanced['alcohol'] * X_advanced['residual sugar']
        X_advanced['acid_total'] = X_advanced['fixed acidity'] + X_advanced['volatile acidity'] + X_advanced['citric acid']
        X_advanced['sulfur_total'] = X_advanced['free sulfur dioxide'] + X_advanced['total sulfur dioxide']
        log_features = ['fixed acidity', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide']
        for feature in log_features:
            X_advanced[f'{feature}_log'] = np.log1p(X_advanced[feature])
        return X_advanced

def predict_quality(input_features):
    model = joblib.load('white_wine_quality_model.pkl')  # Kaydedilen modeli yükleme
    predictor = WineQualityPredictor()
    feature_columns = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
        'pH', 'sulphates', 'alcohol'
    ]
    input_data = pd.DataFrame([input_features], columns=feature_columns)
    advanced_features = predictor.create_advanced_features(input_data)
    prediction = model.predict(advanced_features)
    return prediction[0]

if __name__ == "__main__":
    # Kullanıcıdan giriş verileri liste olarak alınır
    input_list = [
8.6,0.55,0.35,15.55,0.057,35.5,366.5,1.0001,3.04,0.63,11.0
]
    category = predict_quality(input_list)
    print("Tahmin Edilen Kalite Kategorisi:", category)
