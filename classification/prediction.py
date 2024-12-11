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
    model = joblib.load('classification_red_model.pkl')
    print("Model successfully loaded.")
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
    input_list = [
6.9,1.09,0.06,2.1,0.061,12.0,31.0,0.9948,3.51,0.43,11.4
]
    category = predict_quality(input_list)
    print("Estimated Quality Category:", category)
