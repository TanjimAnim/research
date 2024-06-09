import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load your dataset
data = pd.read_csv("pimaTesting.csv")

# Preprocess your data
x = data[
    [
        "Glucose",
        "BloodPressure",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Pregnancies",
    ]
]
y = data["Outcome"]

# Standardize the features
scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    x_scaled,
    y,
    test_size=0.2,
    random_state=42,
)

# Initialize the model with default parameters
rf_model = RandomForestClassifier(random_state=42, criterion="entropy")

# Hyperparameter tuning using GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring="roc_auc",
    verbose=2,
)
grid_search.fit(X_train, y_train)

# Best parameters from GridSearchCV
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train the model with the best parameters
best_rf_model = grid_search.best_estimator_

# Save the model to a file
joblib.dump(best_rf_model, "best_rf_model.pkl")

# Load the model from the file
# best_rf_model = joblib.load('best_rf_model.pkl')

# Make predictions
y_pred = best_rf_model.predict(X_test)
y_pred_prob = best_rf_model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"ROC AUC: {roc_auc}")
print(f"PRECISION: {precision}")
print(f"f1_Score:{f1}")

# Now, load your clinical data which doesn't have an outcome
clinical_data = pd.read_excel("Diabetes (1).xlsx", sheet_name="response")

# Ensure clinical_data has the same features
clinical_features = clinical_data[
    [
        "Glucose",
        "BloodPressure",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Pregnancies",
    ]
]

# Predict the outcomes for the clinical data
clinical_outcome_pred = best_rf_model.predict(clinical_features)
clinical_outcome_pred_prob = best_rf_model.predict_proba(clinical_features)[:, 1]

print(f"clinical_outcome_pred:{clinical_outcome_pred}")
print(f"clinical_outcome_pred_prob:{clinical_outcome_pred_prob}")

# Add predictions to clinical data
clinical_data["PredictedOutcome"] = clinical_outcome_pred
clinical_data["PredictedProbability"] = clinical_outcome_pred_prob

# Save the predictions
clinical_data.to_csv("clinicalData_with_predictions.csv", index=False)
