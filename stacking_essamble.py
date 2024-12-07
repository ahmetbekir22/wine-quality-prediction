# # from sklearn.linear_model import LinearRegression
# # from sklearn.ensemble import StackingRegressor
# # from sklearn.ensemble import RandomForestRegressor
# # import lightgbm as lgb
# # import xgboost as xgb
# # from sklearn.metrics import mean_squared_error, r2_score
# # from sklearn.model_selection import train_test_split
# # import pandas as pd

# # # Veri yükleme
# # data = pd.read_csv('wine_quality_data/winequality-white.csv', sep=';', decimal='.')
# # X = data.drop('quality', axis=1)
# # y = data['quality']

# # # Veriyi bölme
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Base modeller
# # xgboost_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42)
# # lightgbm_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42)
# # randomforest_model = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=42)

# # # Stacking Ensemble
# # estimators = [
# #     ('xgb', xgboost_model),
# #     ('lgbm', lightgbm_model),
# #     ('rf', randomforest_model)
# # ]

# # stacking_model = StackingRegressor(
# #     estimators=estimators,
# #     final_estimator=LinearRegression(),  # Basit model
# #     cv=5
# # )

# # # Modeli eğitme
# # stacking_model.fit(X_train, y_train)

# # # Test seti üzerinde tahminler
# # y_pred = stacking_model.predict(X_test)

# # # Performans metrikleri
# # mse = mean_squared_error(y_test, y_pred)
# # r2 = r2_score(y_test, y_pred)

# # print("\nStacking Ensemble with Linear Regression:")
# # print(f"MSE: {mse:.4f}")
# # print(f"R2 Score: {r2:.4f}")


# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.ensemble import StackingRegressor, RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor
# import lightgbm as lgb
# import xgboost as xgb
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# import pandas as pd

# # Veri yükleme
# data = pd.read_csv('wine_quality_data/winequality-white.csv', sep=';', decimal='.')
# X = data.drop('quality', axis=1)
# y = data['quality']

# # Veriyi bölme
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Base modeller
# xgboost_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42)
# lightgbm_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42)
# randomforest_model = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=42)
# ridge_model = Ridge(alpha=1.0)
# lasso_model = Lasso(alpha=0.1)
# knn_model = KNeighborsRegressor(n_neighbors=5)

# # Stacking Ensemble
# estimators = [
#     ('xgb', xgboost_model),
#     ('lgbm', lightgbm_model),
#     ('rf', randomforest_model),
#     ('ridge', ridge_model),
#     ('lasso', lasso_model),
#     ('knn', knn_model)
# ]

# stacking_model = StackingRegressor(
#     estimators=estimators,
#     final_estimator=LinearRegression(),
#     cv=5
# )

# # Modeli eğitme
# stacking_model.fit(X_train, y_train)

# # Test seti üzerinde tahminler
# y_pred = stacking_model.predict(X_test)

# # Performans metrikleri
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("\nStacking Ensemble with Diversified Models:")
# print(f"MSE: {mse:.4f}")
# print(f"R2 Score: {r2:.4f}")

# Stacking Ensemble with Diversified Models:
# MSE: 0.3778
# R2 Score: 0.5521

import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Veri yükleme
data = pd.read_csv('wine_quality_data/winequality-white.csv', sep=';', decimal='.')
X = data.drop('quality', axis=1)
y = data['quality']

# Veriyi bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optuna optimizasyonu için amaç fonksiyonu
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # 100 - 1000 arasında dene
        'max_depth': trial.suggest_int('max_depth', 3, 15),           # Ağaç derinliği
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),  # Öğrenme oranı
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),      # Alt örnekleme oranı
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Özellik alt kümesi
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),          # L1 cezası
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 5),        # L2 cezası
    }

    # XGBoost modeli
    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)
    
    # Tahmin ve MSE hesaplama
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Optuna optimizasyonu
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# En iyi parametreler
print("Optuna En İyi Parametreler:")
print(study.best_params)

# En iyi model ile tekrar eğitim
best_params = study.best_params
best_model = xgb.XGBRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Performans değerlendirme
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nXGBoost Optuna Optimizasyonu Sonrası Performans:")
print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")
