# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report, accuracy_score
# from data_set_preparing import X_white_train_balanced, y_white_train_balanced, X_white_test_scaled, y_white_test
# from imblearn.over_sampling import SMOTE

# class_weights = {3: 10, 4: 5, 5: 1, 6: 1, 7: 2, 8: 3, 9: 10}

# # Initialize RandomForestClassifier with class weights
# rf_white = RandomForestClassifier(random_state=42, class_weight=class_weights)

# # Set up GridSearchCV for tuning
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [10, 20, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
# }

# grid_search = GridSearchCV(
#     estimator=rf_white,
#     param_grid=param_grid,
#     cv=3,
#     scoring='accuracy',
#     n_jobs=-1,
#     verbose=2
# )

# # Perform grid search with balanced class weights
# grid_search.fit(X_white_train_balanced, y_white_train_balanced)

# # Retrieve the best model
# best_rf_white = grid_search.best_estimator_

# # Evaluate the tuned model
# y_white_pred_tuned = best_rf_white.predict(X_white_test_scaled)
# print("Best Parameters:", grid_search.best_params_)
# print("Tuned White Wine Model Performance:")
# print("Accuracy:", accuracy_score(y_white_test, y_white_pred_tuned))
# print("Classification Report:\n", classification_report(y_white_test, y_white_pred_tuned))


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from data_set_preparing import X_white_train_balanced, y_white_train_balanced, X_white_test_scaled, y_white_test
from imblearn.over_sampling import SMOTE

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None]
}

# Set up class weights for underrepresented classes (adjusted based on your results)
class_weights = {3: 10, 4: 5, 5: 1, 6: 1, 7: 2, 8: 5, 9: 10}

# Initialize RandomForestClassifier with class weights
rf_white = RandomForestClassifier(random_state=42, class_weight=class_weights)

# Apply SMOTE to the training set to balance the data
smote = SMOTE(random_state=42)
X_white_train_balanced_smote, y_white_train_balanced_smote = smote.fit_resample(X_white_train_balanced, y_white_train_balanced)

# Set up GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=rf_white,
    param_grid=param_grid,
    cv=3,  # 3-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all available CPU cores
    verbose=2
)

# Perform grid search with SMOTE-transformed training data
grid_search.fit(X_white_train_balanced_smote, y_white_train_balanced_smote)

# Retrieve the best model after tuning
best_rf_white = grid_search.best_estimator_

# Evaluate the best model on the test set (no SMOTE on test set)
y_white_pred_tuned = best_rf_white.predict(X_white_test_scaled)

# Print the results
print("Best Parameters:", grid_search.best_params_)
print("Tuned White Wine Model Performance:")
print("Accuracy:", accuracy_score(y_white_test, y_white_pred_tuned))
print("Classification Report:\n", classification_report(y_white_test, y_white_pred_tuned))
