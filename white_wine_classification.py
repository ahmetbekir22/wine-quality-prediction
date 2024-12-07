
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler  # Düzeltme yapıldı
# from sklearn.metrics import r2_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.feature_selection import SelectFromModel
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.pipeline import Pipeline


# class WineQualityPredictor:
#     def __init__(self, data_path):
#         # Veriyi yükleme ve ön işleme
#         self.data = pd.read_csv(data_path, sep=';', decimal='.')
#         self.data['quality_category'] = pd.cut(
#             self.data['quality'],
#             bins=[0, 4, 6, 10],  # 3 kategoriye ayırma
#             labels=[0, 1, 2]
#         )

#         # Özellikleri ve hedef değişkenleri ayırma
#         self.X = self.data.drop(['quality', 'quality_category'], axis=1)
#         self.y_reg = self.data['quality']  # Gerçek kaliteyi kullanacağız
#         self.y_cls = self.data['quality_category']

#     def create_advanced_features(self):
#         # Gelişmiş özellik mühendisliği
#         X_advanced = self.X.copy()

#         # Etkileşim özellikleri
#         X_advanced['alcohol_sugar_interaction'] = X_advanced['alcohol'] * X_advanced['residual sugar']
#         X_advanced['acid_total'] = X_advanced['fixed acidity'] + X_advanced['volatile acidity'] + X_advanced['citric acid']
#         X_advanced['sulfur_total'] = X_advanced['free sulfur dioxide'] + X_advanced['total sulfur dioxide']

#         # Log ve kök dönüşümleri
#         log_features = ['fixed acidity', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide']
#         for feature in log_features:
#             X_advanced[f'{feature}_log'] = np.log1p(X_advanced[feature])

#         return X_advanced

#     def train_regression_model(self):
#         # Gelişmiş özellikleri oluşturma
#         X_advanced = self.create_advanced_features()

#         # Veriyi bölme
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_advanced, self.y_reg, test_size=0.2, random_state=42
#         )

#         # Pipeline oluşturma
#         pipeline = Pipeline([
#             ('scaler', StandardScaler()),
#             ('feature_selection', SelectFromModel(RandomForestRegressor(n_estimators=100))),
#             ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
#         ])

#         # Hiperparametre ayarlama
#         param_grid = {
#             'regressor__n_estimators': [100, 200, 300],
#             'regressor__max_depth': [None, 10, 20, 30],
#             'regressor__min_samples_split': [2, 5, 10],
#             'regressor__min_samples_leaf': [1, 2, 4]
#         }

#         # Grid search ile en iyi parametreleri bulma
#         grid_search = GridSearchCV(
#             pipeline,
#             param_grid,
#             cv=5,
#             scoring='neg_mean_squared_error',  # Regresyon problemi olduğu için MSE kullanılır
#             n_jobs=-1
#         )

#         # Modeli eğitme
#         grid_search.fit(X_train, y_train)

#         # En iyi modeli seçme
#         best_model = grid_search.best_estimator_

#         # Tahminler ve performans
#         y_pred = best_model.predict(X_test)

#         print("En İyi Parametreler:", grid_search.best_params_)
#         print("\nRegresyon Performansı:")
#         print("MSE:", mean_squared_error(y_test, y_pred))
#         print("R2 Score:", r2_score(y_test, y_pred))

#         return best_model

# # Modeli çalıştırma
# predictor = WineQualityPredictor('wine_quality_data/winequality-white.csv')  # Dosya yolunu güncelleyin
# reg_model = predictor.train_regression_model()


# import joblib

# # Modeli kaydetme fonksiyonu ekleniyor
# def save_model(model, file_path):
#     joblib.dump(model, file_path)
#     print(f"Model başarıyla kaydedildi: {file_path}")

# # Modeli çalıştırma ve kaydetme
# predictor = WineQualityPredictor('wine_quality_data/winequality-white.csv')  # Dosya yolunu güncelleyin
# reg_model = predictor.train_regression_model()

# # Kaydetme işlemi
# save_model(reg_model, "regression_wine_model.pkl")


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
import joblib

class WineQualityPredictor:
    def __init__(self, data_path):
        # Veriyi yükleme ve ön işleme
        self.data = pd.read_csv(data_path, sep=';', decimal='.')
        self.data['quality_category'] = pd.cut(
            self.data['quality'],
            bins=[0, 4, 6, 10],  # 3 kategoriye ayırma
            labels=[0, 1, 2]
        )

        # Özellikleri ve hedef değişkenleri ayırma
        self.X = self.data.drop(['quality', 'quality_category'], axis=1)
        self.y_reg = self.data['quality']  # Gerçek kaliteyi kullanacağız
        self.y_cls = self.data['quality_category']

    def create_advanced_features(self):
        # Gelişmiş özellik mühendisliği
        X_advanced = self.X.copy()

        # Etkileşim özellikleri
        X_advanced['alcohol_sugar_interaction'] = X_advanced['alcohol'] * X_advanced['residual sugar']
        X_advanced['acid_total'] = X_advanced['fixed acidity'] + X_advanced['volatile acidity'] + X_advanced['citric acid']
        X_advanced['sulfur_total'] = X_advanced['free sulfur dioxide'] + X_advanced['total sulfur dioxide']

        # Log dönüşümleri
        log_features = ['fixed acidity', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide']
        for feature in log_features:
            X_advanced[f'{feature}_log'] = np.log1p(X_advanced[feature])

        return X_advanced

    def train_regression_model(self):
        # Gelişmiş özellikleri oluşturma
        X_advanced = self.create_advanced_features()

        # Veriyi bölme
        X_train, X_test, y_train, y_test = train_test_split(
            X_advanced, self.y_reg, test_size=0.2, random_state=42
        )

        # Pipeline oluşturma
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(RandomForestRegressor(n_estimators=100))),
            ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
        ])

        # Hiperparametre ayarlama
        param_grid = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [None, 10, 20, 30],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4]
        }

        # Grid search ile en iyi parametreleri bulma
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        # Modeli eğitme
        grid_search.fit(X_train, y_train)

        # En iyi modeli seçme
        best_model = grid_search.best_estimator_

        # Tahminler ve performans
        y_pred = best_model.predict(X_test)

        print("En İyi Parametreler:", grid_search.best_params_)
        print("\nRegresyon Performansı:")
        print("MSE:", mean_squared_error(y_test, y_pred))
        print("R2 Score:", r2_score(y_test, y_pred))

        return best_model

    def save_model(self, model, file_path):
        # Modeli kaydetme
        joblib.dump(model, file_path)
        print(f"Model başarıyla kaydedildi: {file_path}")

# Modeli çalıştırma
predictor = WineQualityPredictor('wine_quality_data/winequality-white.csv')  # Dosya yolunu güncelleyin
reg_model = predictor.train_regression_model()

# Modeli kaydetme
predictor.save_model(reg_model, "regression_wine_model.pkl")
