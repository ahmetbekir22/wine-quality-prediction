

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from data_set_preparing import X_white_train_balanced, y_white_train_balanced, X_white_test_scaled, y_white_test
from imblearn.over_sampling import SMOTE

# Hiperparametre ızgarası
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],  
}

# Sınıf ağırlıkları (düşük temsil edilen sınıflara daha fazla ağırlık)
#class_weights = {3: 10, 4: 5, 5: 1, 6: 1, 7: 2, 8: 5, 9: 10}
#class_weights = {3: 15, 4: 10, 5: 5, 6: 1, 7: 5, 8: 10, 9: 15}
#class_weights = {3: 15, 4: 20, 5: 5, 6: 1, 7: 5, 8: 15, 9: 20}
#class_weights = {3: 20, 4: 25, 5: 10, 6: 3, 7: 5, 8: 15, 9: 20}
class_weights = {3: 15, 4: 25, 5: 10, 6: 3, 7: 5, 8: 15, 9: 20}

# Random Forest modelini başlatma
rf_white = RandomForestClassifier(random_state=42, class_weight=class_weights)

# GridSearchCV ile hiperparametre optimizasyonu
grid_search = GridSearchCV(
    estimator=rf_white,
    param_grid=param_grid,
    cv=3,  # 3-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Tüm işlemcileri kullan
    verbose=2
)

# SMOTE ile eğitim setine veri artırımı uygulayalım
smote = SMOTE(random_state=42)
X_white_train_balanced_smote, y_white_train_balanced_smote = smote.fit_resample(X_white_train_balanced, y_white_train_balanced)

# GridSearchCV ile modelin parametrelerini optimize edelim
grid_search.fit(X_white_train_balanced_smote, y_white_train_balanced_smote)

# En iyi modeli alalım
best_rf_white = grid_search.best_estimator_

# Test seti üzerinde modelin performansını değerlendirelim
y_white_pred_tuned = best_rf_white.predict(X_white_test_scaled)

# En iyi parametreleri ve model performansını yazdıralım
print("Best Parameters:", grid_search.best_params_)
print("Tuned White Wine Model Performance:")
print("Accuracy:", accuracy_score(y_white_test, y_white_pred_tuned))
print("Classification Report:\n", classification_report(y_white_test, y_white_pred_tuned))

# Modelin doğruluğunu değerlendirmek
rf_white = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_white.fit(X_white_train_balanced, y_white_train_balanced)

y_white_pred = rf_white.predict(X_white_test_scaled)

print("White Wine Model Performance:")
print("Accuracy:", accuracy_score(y_white_test, y_white_pred))
print("Classification Report:\n", classification_report(y_white_test, y_white_pred))


# from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report, accuracy_score
# from data_set_preparing import X_white_train_balanced, y_white_train_balanced, X_white_test_scaled, y_white_test
# from imblearn.over_sampling import SMOTE

# # Hiperparametre ızgarası
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [10, 20, None],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'min_child_weight': [1, 5, 10],
#     'subsample': [0.7, 0.8, 1.0],
#     'colsample_bytree': [0.7, 0.8, 1.0]
# }

# # Dengesiz sınıflara daha fazla ağırlık vermek için scale_pos_weight parametresi kullanacağız
# scale_pos_weight = 15  # Düşük temsil edilen sınıflara daha fazla ağırlık

# # XGBoost modelini başlatma
# xgb_white = XGBClassifier(
#     random_state=42,
#     scale_pos_weight=scale_pos_weight,
#     n_estimators=200,
#     max_depth=20,
#     learning_rate=0.1
# )

# # GridSearchCV ile hiperparametre optimizasyonu
# grid_search = GridSearchCV(
#     estimator=xgb_white,
#     param_grid=param_grid,
#     cv=3,  # 3-fold cross-validation
#     scoring='accuracy',
#     n_jobs=-1,  # Tüm işlemcileri kullan
#     verbose=2
# )

# # SMOTE ile eğitim setine veri artırımı uygulayalım
# smote = SMOTE(random_state=42)
# X_white_train_balanced_smote, y_white_train_balanced_smote = smote.fit_resample(X_white_train_balanced, y_white_train_balanced)

# # GridSearchCV ile modelin parametrelerini optimize edelim
# grid_search.fit(X_white_train_balanced_smote, y_white_train_balanced_smote)

# # En iyi modeli alalım
# best_xgb_white = grid_search.best_estimator_

# # Test seti üzerinde modelin performansını değerlendirelim
# y_white_pred_tuned = best_xgb_white.predict(X_white_test_scaled)

# # En iyi parametreleri ve model performansını yazdıralım
# print("Best Parameters:", grid_search.best_params_)
# print("Tuned XGBoost White Wine Model Performance:")
# print("Accuracy:", accuracy_score(y_white_test, y_white_pred_tuned))
# print("Classification Report:\n", classification_report(y_white_test, y_white_pred_tuned))

# # Modelin doğruluğunu değerlendirmek
# xgb_white = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
# xgb_white.fit(X_white_train_balanced, y_white_train_balanced)

# y_white_pred = xgb_white.predict(X_white_test_scaled)

# print("XGBoost White Wine Model Performance:")
# print("Accuracy:", accuracy_score(y_white_test, y_white_pred))
# print("Classification Report:\n", classification_report(y_white_test, y_white_pred))
