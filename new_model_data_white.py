### this is the model for white wine 
### accurecy score is 0.917
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Veriyi yükleyin ve hedef ile özellikleri ayırın
wine_data = pd.read_csv('wine_quality_data/winequality-white.csv', sep=';')
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Veriyi eğitim ve test setlerine ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri standardize edelim
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest modelini tanımlayalım ve eğitelim
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Test seti üzerinde tahmin yapalım
y_pred = model.predict(X_test_scaled)

# Modelin performansını değerlendirelim
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

accuracy = (abs(y_pred - y_test) < 1).mean()  # Tahminin hedef değere yakın olmasını kontrol et


# Sonuçları yazdıralım
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)
print("Accuracy Score:", accuracy)



# Feature Importance'ı görselleştirelim
feature_importances = model.feature_importances_

# Özellikleri liste halinde alalım
features = X.columns

# Feature importance'ı bir DataFrame'e dökelim
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

# Önem sırasına göre sıralayalım
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Görselleştirme
plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance (Random Forest)')
plt.gca().invert_yaxis()  # En önemli özelliği üstte göstermek için
plt.show()


