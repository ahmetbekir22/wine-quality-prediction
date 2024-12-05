# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report, accuracy_score
# from data_set_preparing import X_white_train_balanced, y_white_train_balanced, X_white_test_scaled, y_white_test
# from imblearn.over_sampling import SMOTE

# # Define the parameter grid
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [10, 20, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],  
# }

# # Set up class weights
# class_weights = {3: 5, 4: 3, 5: 1, 6: 1, 7: 2, 8: 3, 9: 5}
# #class_weights = {3: 10, 4: 5, 5: 1, 6: 1, 7: 2, 8: 5, 9: 10}

# # Initialize the Random Forest classifier
# rf_white = RandomForestClassifier(random_state=42, class_weight=class_weights)

# # Set up GridSearchCV
# grid_search = GridSearchCV(
#     estimator=rf_white,
#     param_grid=param_grid,
#     cv=3,  # 3-fold cross-validation
#     scoring='accuracy',
#     n_jobs=-1,  # Use all available CPU cores
#     verbose=2
# )

# # Perform the grid search
# grid_search.fit(X_white_train_balanced, y_white_train_balanced)

# # Retrieve the best model
# best_rf_white = grid_search.best_estimator_

# # Evaluate the best model on the test set
# y_white_pred_tuned = best_rf_white.predict(X_white_test_scaled)
# print("Best Parameters:", grid_search.best_params_)
# print("Tuned White Wine Model Performance:")
# print("Accuracy:", accuracy_score(y_white_test, y_white_pred_tuned))
# print("Classification Report:\n", classification_report(y_white_test, y_white_pred_tuned))


# # Apply SMOTE to the test set with adjusted k_neighbors
# smote_test = SMOTE(random_state=42, k_neighbors=3)
# X_white_test_balanced, y_white_test_balanced = smote_test.fit_resample(
#     X_white_test_scaled, y_white_test
# )

# # Evaluate the trained model on the balanced test set
# y_white_pred_balanced = best_rf_white.predict(X_white_test_balanced)

# # Print evaluation metrics
# print("Evaluation on Balanced Test Set:")
# print("Accuracy:", accuracy_score(y_white_test_balanced, y_white_pred_balanced))
# print("Classification Report:\n", classification_report(y_white_test_balanced, y_white_pred_balanced))



###################
# # import pandas as pd
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import classification_report, accuracy_score
# # from data_set_preparing import X_white_train_balanced, y_white_train_balanced, X_white_test_scaled, y_white_test

# # # Original feature names
# # original_features = ['fixed acidity', 'volatile acidity', 'citric acid', 
# #                      'residual sugar', 'chlorides', 'free sulfur dioxide', 
# #                      'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

# # # Yeni bir özellik seti oluşturuyoruz (chlorides, pH, free sulfur dioxide çıkarılıyor)
# # selected_features = ['fixed acidity', 'volatile acidity', 'citric acid', 
# #                      'residual sugar', 'total sulfur dioxide', 
# #                      'density', 'sulphates', 'alcohol']

# # # Convert NumPy arrays back to pandas DataFrames with column names
# # X_white_train_balanced_df = pd.DataFrame(X_white_train_balanced, columns=original_features)
# # X_white_test_scaled_df = pd.DataFrame(X_white_test_scaled, columns=original_features)

# # # Select only the desired features
# # X_white_train_selected = X_white_train_balanced_df[selected_features]
# # X_white_test_selected = X_white_test_scaled_df[selected_features]

# # # İlk modeli, tüm özelliklerle eğitiyoruz
# # rf_all_features = RandomForestClassifier(random_state=42, class_weight="balanced")
# # rf_all_features.fit(X_white_train_balanced_df, y_white_train_balanced)
# # y_pred_all = rf_all_features.predict(X_white_test_scaled_df)

# # # Özellik seçiminden sonra modeli eğitiyoruz
# # rf_selected_features = RandomForestClassifier(random_state=42, class_weight="balanced")
# # rf_selected_features.fit(X_white_train_selected, y_white_train_balanced)
# # y_pred_selected = rf_selected_features.predict(X_white_test_selected)

# # # Performansları karşılaştırıyoruz
# # print("Model Performansı (Tüm Özellikler):")
# # print("Accuracy:", accuracy_score(y_white_test, y_pred_all))
# # print("Classification Report:\n", classification_report(y_white_test, y_pred_all))

# # print("\nModel Performansı (Seçilmiş Özellikler):")
# # print("Accuracy:", accuracy_score(y_white_test, y_pred_selected))
# # print("Classification Report:\n", classification_report(y_white_test, y_pred_selected))



# # import matplotlib.pyplot as plt

# # # Tüm özelliklerdeki önem
# # feature_importance_all = rf_all_features.feature_importances_
# # plt.barh(original_features, feature_importance_all)
# # plt.title("Feature Importance (Tüm Özellikler)")
# # plt.show()

# # # Seçilmiş özelliklerdeki önem
# # feature_importance_selected = rf_selected_features.feature_importances_
# # plt.barh(selected_features, feature_importance_selected)
# # plt.title("Feature Importance (Seçilmiş Özellikler)")
# # plt.show()


# from imblearn.over_sampling import SMOTE
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
# import pandas as pd

# # SMOTE uygulaması
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_white_train_balanced, y_white_train_balanced)

# # Yeni dengelenmiş veri seti boyutlarını kontrol et
# print("SMOTE sonrası eğitim seti sınıf dağılımı:")
# print(pd.Series(y_train_smote).value_counts())

# # Modeli eğitme (SMOTE sonrası)
# rf_smote = RandomForestClassifier(random_state=42, class_weight="balanced")
# rf_smote.fit(X_train_smote, y_train_smote)

# # Test setinde tahmin yapma
# y_pred_smote = rf_smote.predict(X_white_test_scaled)

# # Performansı değerlendirme
# print("\nSMOTE ile Denge Sağlanmış Model Performansı:")
# print("Accuracy:", accuracy_score(y_white_test, y_pred_smote))
# print("Classification Report:\n", classification_report(y_white_test, y_pred_smote))


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report, accuracy_score
# from data_set_preparing import X_white_train_balanced, y_white_train_balanced, X_white_test_scaled, y_white_test
# from imblearn.over_sampling import SMOTE

# # Define the parameter grid
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [10, 20, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],  
# }

# # Set up class weights
# class_weights = {3: 5, 4: 3, 5: 1, 6: 1, 7: 2, 8: 3, 9: 5}

# # Initialize the Random Forest classifier
# rf_white = RandomForestClassifier(random_state=42, class_weight=class_weights)

# # Set up GridSearchCV
# grid_search = GridSearchCV(
#     estimator=rf_white,
#     param_grid=param_grid,
#     cv=3,  # 3-fold cross-validation
#     scoring='accuracy',
#     n_jobs=-1,  # Use all available CPU cores
#     verbose=2
# )

# # Apply SMOTE to the training set
# smote = SMOTE(random_state=42)
# X_white_train_balanced_smote, y_white_train_balanced_smote = smote.fit_resample(X_white_train_balanced, y_white_train_balanced)

# # Perform the grid search with the SMOTE transformed training set
# grid_search.fit(X_white_train_balanced_smote, y_white_train_balanced_smote)

# # Retrieve the best model
# best_rf_white = grid_search.best_estimator_

# # Evaluate the best model on the test set (no SMOTE on test set)
# y_white_pred_tuned = best_rf_white.predict(X_white_test_scaled)
# print("Best Parameters:", grid_search.best_params_)
# print("Tuned White Wine Model Performance:")
# print("Accuracy:", accuracy_score(y_white_test, y_white_pred_tuned))
# print("Classification Report:\n", classification_report(y_white_test, y_white_pred_tuned))


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score

# # Random Forest modelini başlatıyoruz
# rf_white = RandomForestClassifier(random_state=42, class_weight='balanced')

# # Modeli SMOTE ile dengelenmiş eğitim verisi üzerinde eğitiyoruz
# rf_white.fit(X_white_train_balanced, y_white_train_balanced)

# # Test seti üzerinde tahmin yapıyoruz
# y_white_pred = rf_white.predict(X_white_test_scaled)

# # Modelin performansını değerlendiriyoruz
# print("White Wine Model Performance:")
# print("Accuracy:", accuracy_score(y_white_test, y_white_pred))
# print("Classification Report:\n", classification_report(y_white_test, y_white_pred))


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report, accuracy_score
# from data_set_preparing import X_white_train_balanced, y_white_train_balanced, X_white_test_scaled, y_white_test
# from imblearn.over_sampling import SMOTE

# # Hiperparametre ızgarası
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [10, 20, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],  
# }

# # Sınıf ağırlıkları (düşük temsil edilen sınıflara daha fazla ağırlık)
# #class_weights = {3: 10, 4: 5, 5: 1, 6: 1, 7: 2, 8: 5, 9: 10}
# #class_weights = {3: 15, 4: 10, 5: 5, 6: 1, 7: 5, 8: 10, 9: 15}
# #class_weights = {3: 15, 4: 20, 5: 5, 6: 1, 7: 5, 8: 15, 9: 20}
# #class_weights = {3: 20, 4: 25, 5: 10, 6: 3, 7: 5, 8: 15, 9: 20}
# class_weights = {3: 15, 4: 25, 5: 10, 6: 3, 7: 5, 8: 15, 9: 20}

# # Random Forest modelini başlatma
# rf_white = RandomForestClassifier(random_state=42, class_weight=class_weights)

# # GridSearchCV ile hiperparametre optimizasyonu
# grid_search = GridSearchCV(
#     estimator=rf_white,
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
# best_rf_white = grid_search.best_estimator_

# # Test seti üzerinde modelin performansını değerlendirelim
# y_white_pred_tuned = best_rf_white.predict(X_white_test_scaled)

# # En iyi parametreleri ve model performansını yazdıralım
# print("Best Parameters:", grid_search.best_params_)
# print("Tuned White Wine Model Performance:")
# print("Accuracy:", accuracy_score(y_white_test, y_white_pred_tuned))
# print("Classification Report:\n", classification_report(y_white_test, y_white_pred_tuned))

# # Modelin doğruluğunu değerlendirmek
# rf_white = RandomForestClassifier(random_state=42, class_weight='balanced')
# rf_white.fit(X_white_train_balanced, y_white_train_balanced)

# y_white_pred = rf_white.predict(X_white_test_scaled)

# print("White Wine Model Performance:")
# print("Accuracy:", accuracy_score(y_white_test, y_white_pred))
# print("Classification Report:\n", classification_report(y_white_test, y_white_pred))


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from data_set_preparing import X_white_train_balanced, y_white_train_balanced, X_white_test_scaled, y_white_test
from imblearn.over_sampling import SMOTE

# Hiperparametre ızgarası
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# Dengesiz sınıflara daha fazla ağırlık vermek için scale_pos_weight parametresi kullanacağız
scale_pos_weight = 15  # Düşük temsil edilen sınıflara daha fazla ağırlık

# XGBoost modelini başlatma
xgb_white = XGBClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    n_estimators=200,
    max_depth=20,
    learning_rate=0.1
)

# GridSearchCV ile hiperparametre optimizasyonu
grid_search = GridSearchCV(
    estimator=xgb_white,
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
best_xgb_white = grid_search.best_estimator_

# Test seti üzerinde modelin performansını değerlendirelim
y_white_pred_tuned = best_xgb_white.predict(X_white_test_scaled)

# En iyi parametreleri ve model performansını yazdıralım
print("Best Parameters:", grid_search.best_params_)
print("Tuned XGBoost White Wine Model Performance:")
print("Accuracy:", accuracy_score(y_white_test, y_white_pred_tuned))
print("Classification Report:\n", classification_report(y_white_test, y_white_pred_tuned))

# Modelin doğruluğunu değerlendirmek
xgb_white = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
xgb_white.fit(X_white_train_balanced, y_white_train_balanced)

y_white_pred = xgb_white.predict(X_white_test_scaled)

print("XGBoost White Wine Model Performance:")
print("Accuracy:", accuracy_score(y_white_test, y_white_pred))
print("Classification Report:\n", classification_report(y_white_test, y_white_pred))
