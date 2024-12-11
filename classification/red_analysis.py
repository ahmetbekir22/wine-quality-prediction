import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Model kaydetme/yükleme için
import matplotlib.pyplot as plt



class WineQualityPredictor:

    def plot_feature_importances(model, feature_names):
    # Özellik önem sırasını al
        importances = model.named_steps['feature_selection'].estimator_.feature_importances_
        indices = np.argsort(importances)[::-1]  # Önem sırasına göre tersten sırala
        sorted_features = [feature_names[i] for i in indices]

        # Yatay çubuk grafik oluştur
        plt.figure(figsize=(8, 10))
        plt.title("Özelliklerin Önem Sırası")
        plt.barh(range(len(importances)), importances[indices], align="center")
        plt.yticks(range(len(importances)), sorted_features)
        plt.gca().invert_yaxis()  # En önemli özelliğin üstte olması için ekseni ters çevir
        plt.xlabel("Önem Derecesi")
        plt.tight_layout()
        plt.show()

    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, sep=';', decimal='.')
        self.label_encoder = LabelEncoder()
        self.data['quality_category'] = pd.cut(
            self.data['quality'], 
            bins=[0, 4, 6, 10],  
            labels=["Low Quality", "Medium Quality", "High Quality"]
        )
        self.X = self.data.drop(['quality', 'quality_category'], axis=1)
        self.y_cls = self.data['quality_category']
        
    def create_advanced_features(self, X):
        X_advanced = X.copy()
        X_advanced['alcohol_sugar_interaction'] = X_advanced['alcohol'] * X_advanced['residual sugar']
        X_advanced['acid_total'] = X_advanced['fixed acidity'] + X_advanced['volatile acidity'] + X_advanced['citric acid']
        X_advanced['sulfur_total'] = X_advanced['free sulfur dioxide'] + X_advanced['total sulfur dioxide']
        log_features = ['fixed acidity', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide']
        for feature in log_features:
            X_advanced[f'{feature}_log'] = np.log1p(X_advanced[feature])
        return X_advanced
    
    def train_classification_model(self):
        X_advanced = self.create_advanced_features(self.X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_advanced, self.y_cls, test_size=0.2, random_state=42
        )
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100))),
            ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
        ])
        param_grid = {
            'classifier__n_estimators': [100],
            'classifier__max_depth': [30],
            'classifier__min_samples_split': [5],
            'classifier__min_samples_leaf': [1]
        }
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        print("En İyi Parametreler:", grid_search.best_params_)
        print("\nAccuracy:", accuracy_score(y_test, y_pred))
        joblib.dump(best_model, 'red_wine_quality_model.pkl')  # Modeli kaydetme
        return best_model

if __name__ == "__main__":
    predictor = WineQualityPredictor('wine_quality_data/winequality-red.csv')  # Dosya yolunu güncelleyin
    predictor.train_classification_model()
    best_model = predictor.train_classification_model()  # En iyi modeli al
    
    # Özellik isimlerini al ve önem sırasını çiz
    feature_names = predictor.create_advanced_features(predictor.X).columns
    WineQualityPredictor.plot_feature_importances(best_model, feature_names)
