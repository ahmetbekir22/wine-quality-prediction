import joblib  # For saving the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from data_set_preparing import X_red_train_balanced, y_red_train_balanced, X_red_test_scaled, y_red_test

# Train the model
rf_red = RandomForestClassifier(random_state=42)
rf_red.fit(X_red_train_balanced, y_red_train_balanced)

# Make predictions
y_red_pred = rf_red.predict(X_red_test_scaled)

# Evaluate the model
print("Red Wine Model Performance:")
print("Accuracy:", accuracy_score(y_red_test, y_red_pred))
print("Classification Report:\n", classification_report(y_red_test, y_red_pred))

# Save the model
joblib.dump(rf_red, "red_wine_model.pkl")
