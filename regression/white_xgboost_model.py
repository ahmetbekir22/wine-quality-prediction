import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Veri yükleme
data = pd.read_csv('wine_quality_data/winequality-white.csv', sep=';', decimal='.')
X = data.drop('quality', axis=1)
y = data['quality']

# Veriyi bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test setini kaydetme
test_data = X_test.copy()
test_data['quality'] = y_test  # Hedef sütunu ekleme
test_data.to_csv("test_data_white_wine.csv", index=False)
print("Test verisi başarıyla kaydedildi: test_data_wine.csv")

# Optuna tarafından optimize edilen XGBoost parametreleri
optimized_params = {
    'n_estimators': 449,
    'max_depth': 11,
    'learning_rate': 0.0309,
    'subsample': 0.7449,
    'colsample_bytree': 0.7765,
    'reg_alpha': 0.3222,  # L1 cezası
    'reg_lambda': 1.5832, # L2 cezası
    'random_state': 42
}

# XGBoost modeli oluşturma
model = xgb.XGBRegressor(**optimized_params)

# Modeli eğitme
model.fit(X_train, y_train)

# Test seti üzerinde tahminler
y_pred = model.predict(X_test)

# Performans metrikleri
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performansı:")
print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Modeli kaydetme
joblib.dump(model, "xgboost_white_model.pkl")
print("Model başarıyla kaydedildi: xgboost_wine_model.pkl")
