# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.feature_selection import SelectFromModel
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

#     def create_advanced_features(self):
#         # Gelişmiş özellik mühendisliğif
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

#         # Hiperparametre ayarlama (daha geniş bir aralık)
#         param_grid = {
#             'regressor__n_estimators': [100, 200, 300, 500],
#             'regressor__max_depth': [None, 10, 20, 30, 50],
#             'regressor__min_samples_split': [2, 5, 10, 20],
#             'regressor__min_samples_leaf': [1, 2, 4, 8]
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


# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.pipeline import Pipeline
# import xgboost as xgb

# class WineQualityPredictor:
#     def __init__(self, data_path):
#         self.data = pd.read_csv(data_path, sep=';', decimal='.')
#         self.data['quality_category'] = pd.cut(
#             self.data['quality'],
#             bins=[0, 4, 6, 10],
#             labels=[0, 1, 2]
#         )
#         self.X = self.data.drop(['quality', 'quality_category'], axis=1)
#         self.y_reg = self.data['quality']

#     def create_advanced_features(self):
#         X_advanced = self.X.copy()
#         X_advanced['alcohol_sugar_interaction'] = X_advanced['alcohol'] * X_advanced['residual sugar']
#         X_advanced['acid_total'] = X_advanced['fixed acidity'] + X_advanced['volatile acidity'] + X_advanced['citric acid']
#         X_advanced['sulfur_total'] = X_advanced['free sulfur dioxide'] + X_advanced['total sulfur dioxide']
#         log_features = ['fixed acidity', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide']
#         for feature in log_features:
#             X_advanced[f'{feature}_log'] = np.log1p(X_advanced[feature])
#         return X_advanced

# class WineQualityPredictorImproved(WineQualityPredictor):
#     def train_xgboost_model(self):
#         X_advanced = self.create_advanced_features()
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_advanced, self.y_reg, test_size=0.2, random_state=42
#         )
#         pipeline = Pipeline([
#             ('scaler', StandardScaler()),
#             ('regressor', xgb.XGBRegressor(random_state=42))
#         ])
#         param_distributions = {
#             'regressor__n_estimators': [100, 200, 300, 500, 700],
#             'regressor__max_depth': [3, 5, 7, 10, 15],
#             'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
#             'regressor__subsample': [0.7, 0.8, 0.9, 1.0],
#             'regressor__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
#         }
#         random_search = RandomizedSearchCV(
#             pipeline,
#             param_distributions,
#             n_iter=20,
#             cv=5,
#             scoring='neg_mean_squared_error',
#             n_jobs=-1,
#             random_state=42
#         )
#         random_search.fit(X_train, y_train)
#         best_model = random_search.best_estimator_
#         y_pred = best_model.predict(X_test)
#         print("En İyi Parametreler:", random_search.best_params_)
#         print("MSE:", mean_squared_error(y_test, y_pred))
#         print("R2 Score:", r2_score(y_test, y_pred))
#         return best_model

# # Dosya yolunu güncelleyin
# data_path = 'wine_quality_data/winequality-white.csv'
# predictor = WineQualityPredictorImproved(data_path)
# best_model = predictor.train_xgboost_model()


# En İyi Parametreler: {'regressor__subsample': 0.7, 'regressor__n_estimators': 300, 'regressor__max_depth': 10, 'regressor__learning_rate': 0.05, 'regressor__colsample_bytree': 0.7}
# MSE: 0.34602362743002635
# R2 Score: 0.5896994707088217

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.pipeline import Pipeline
# import xgboost as xgb

# class WineQualityPredictor:
#     def __init__(self, data_path):
#         # Veri yükleme ve ön işleme
#         self.data = pd.read_csv(data_path, sep=';', decimal='.')
#         self.data['quality_category'] = pd.cut(
#             self.data['quality'],
#             bins=[0, 4, 6, 10],
#             labels=[0, 1, 2]
#         )
#         self.X = self.data.drop(['quality', 'quality_category'], axis=1)
#         self.y_reg = self.data['quality']

#     def create_advanced_features(self):
#         # Özellik mühendisliği
#         X_advanced = self.X.copy()
#         X_advanced['alcohol_sugar_interaction'] = X_advanced['alcohol'] * X_advanced['residual sugar']
#         X_advanced['acid_total'] = X_advanced['fixed acidity'] + X_advanced['volatile acidity'] + X_advanced['citric acid']
#         X_advanced['sulfur_total'] = X_advanced['free sulfur dioxide'] + X_advanced['total sulfur dioxide']
        
#         # Log dönüşümleri
#         log_features = ['fixed acidity', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide']
#         for feature in log_features:
#             X_advanced[f'{feature}_log'] = np.log1p(X_advanced[feature])
#         return X_advanced

#     def train_xgboost_model(self):
#         # Gelişmiş özellik oluşturma
#         X_advanced = self.create_advanced_features()
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_advanced, self.y_reg, test_size=0.2, random_state=42
#         )
        
#         # Pipeline oluşturma
#         pipeline = Pipeline([
#             ('scaler', StandardScaler()),
#             ('regressor', xgb.XGBRegressor(random_state=42))
#         ])
        
#         # Hiperparametre aralığı
#         param_distributions = {
#             'regressor__n_estimators': [100, 200, 300, 500],
#             'regressor__max_depth': [3, 5, 7, 10, 15],
#             'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
#             'regressor__subsample': [0.7, 0.8, 0.9, 1.0],
#             'regressor__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
#             'regressor__reg_alpha': [0, 0.1, 0.5, 1],  # L1 cezası
#             'regressor__reg_lambda': [1, 1.5, 2, 5]   # L2 cezası
#         }
        
#         # RandomizedSearchCV
#         random_search = RandomizedSearchCV(
#             pipeline,
#             param_distributions,
#             n_iter=30,  # Daha hızlı optimizasyon için iterasyon sayısı
#             cv=5,
#             scoring='neg_mean_squared_error',
#             n_jobs=-1,
#             random_state=42
#         )
        
#         # Model eğitme
#         random_search.fit(X_train, y_train)
#         best_model = random_search.best_estimator_
        
#         # Test seti tahminleri
#         y_pred = best_model.predict(X_test)
        
#         # Performans sonuçları
#         print("En İyi Parametreler:", random_search.best_params_)
#         print("\nModel Performansı:")
#         print("MSE:", mean_squared_error(y_test, y_pred))
#         print("R2 Score:", r2_score(y_test, y_pred))
        
#         # Feature Importance
#         feature_importances = best_model.named_steps['regressor'].feature_importances_
#         features = X_advanced.columns
#         sorted_features = sorted(zip(features, feature_importances), key=lambda x: x[1], reverse=True)
        
#         print("\nÖzelliklerin Önem Sıralaması:")
#         for feature, importance in sorted_features:
#             print(f"{feature}: {importance:.4f}")
        
#         return best_model

# # Modeli çalıştırma
# data_path = 'wine_quality_data/winequality-white.csv'  # Dosya yolunu güncelleyin
# predictor = WineQualityPredictor(data_path)
# best_model = predictor.train_xgboost_model()

# En İyi Parametreler: {'regressor__subsample': 0.8, 'regressor__reg_lambda': 1, 'regressor__reg_alpha': 0, 'regressor__n_estimators': 300, 'regressor__max_depth': 15, 'regressor__learning_rate': 0.05, 'regressor__colsample_bytree': 0.7}

# Model Performansı:
# MSE: 0.3482368678691935
# R2 Score: 0.5870751015858706

# Özelliklerin Önem Sıralaması:
# alcohol: 0.1965
# free sulfur dioxide_log: 0.1211
# density: 0.0602
# alcohol_sugar_interaction: 0.0597
# sulfur_total: 0.0517
# fixed acidity_log: 0.0488
# free sulfur dioxide: 0.0462
# acid_total: 0.0450
# sulphates: 0.0422
# pH: 0.0419
# total sulfur dioxide: 0.0415
# chlorides: 0.0392
# residual sugar: 0.0350
# residual sugar_log: 0.0334
# volatile acidity: 0.0324
# chlorides_log: 0.0300
# citric acid: 0.0293
# total sulfur dioxide_log: 0.0285
# fixed acidity: 0.0174

# from sklearn.ensemble import StackingRegressor
# from sklearn.ensemble import RandomForestRegressor
# import lightgbm as lgb
# import xgboost as xgb
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import pandas as pd

# # Veri yükleme ve hazırlık
# data = pd.read_csv('wine_quality_data/winequality-white.csv', sep=';', decimal='.')
# X = data.drop('quality', axis=1)
# y = data['quality']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model tanımları
# estimators = [
#     ('xgb', xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=10, random_state=42)),
#     ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
#     ('lgbm', lgb.LGBMRegressor(n_estimators=300, random_state=42))
# ]

# # Stacking model
# stacking_model = StackingRegressor(
#     estimators=estimators,
#     final_estimator=xgb.XGBRegressor(learning_rate=0.01, n_estimators=100, random_state=42)
# )

# # Modeli eğitme ve test
# stacking_model.fit(X_train, y_train)
# y_pred = stacking_model.predict(X_test)

# print("MSE:", mean_squared_error(y_test, y_pred))
# print("R2 Score:", r2_score(y_test, y_pred))


# MSE: 0.4361336164154505
# R2 Score: 0.4828507666196238


# import optuna
# import xgboost as xgb
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# import pandas as pd

# # Veri yükleme
# data = pd.read_csv('wine_quality_data/winequality-white.csv', sep=';', decimal='.')
# X = data.drop('quality', axis=1)
# y = data['quality']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 500),
#         'max_depth': trial.suggest_int('max_depth', 5, 20),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
#         'subsample': trial.suggest_float('subsample', 0.7, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1, 5)
#     }
#     model = xgb.XGBRegressor(**params, random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     return mean_squared_error(y_test, y_pred)

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=30)

# print("En iyi parametreler:", study.best_params)

#En iyi parametreler: {'n_estimators': 449, 'max_depth': 11,
# 'learning_rate': 0.030955410335732618, 'subsample': 0.7449112121999851,
# 'colsample_bytree': 0.7765863007912609, 'reg_alpha': 0.3221963151727452, 
# 'reg_lambda': 1.5832044188693135}

# import xgboost as xgb
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# import pandas as pd

# # Veri yükleme
# data = pd.read_csv('wine_quality_data/winequality-white.csv', sep=';', decimal='.')
# X = data.drop('quality', axis=1)
# y = data['quality']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # XGBoost modeli
# model = xgb.XGBRegressor(
#     n_estimators=1000,
#     learning_rate=0.05,
#     max_depth=10,
#     early_stopping_rounds=50
# )

# # Modeli eğitme
# model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
# y_pred = model.predict(X_test)

# print("MSE:", mean_squared_error(y_test, y_pred))
# print("R2 Score:", r2_score(y_test, y_pred))


# MSE: 0.3698358400674221
# R2 Score: 0.5614639322246953



# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# import pandas as pd

# # Veri yükleme
# data = pd.read_csv('wine_quality_data/winequality-white.csv', sep=';', decimal='.')
# X = data.drop('quality', axis=1)
# y = data['quality']

# # Polinomsal özellikler
# poly = PolynomialFeatures(degree=2, interaction_only=True)
# X_poly = poly.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# # Lineer model
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print("MSE:", mean_squared_error(y_test, y_pred))
# print("R2 Score:", r2_score(y_test, y_pred))

# MSE: 0.6350381723934942
# R2 Score: 0.24699795736970742


import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Veri yükleme
data = pd.read_csv('wine_quality_data/winequality-white.csv', sep=';', decimal='.')
X = data.drop('quality', axis=1)
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM modeli
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=15)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
