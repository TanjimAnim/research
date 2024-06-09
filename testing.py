import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


# Load your dataset
data = pd.read_csv("pimaTesting.csv")

# Preprocess your data
# Assume 'Variant' is the feature and 'Diabetes' is the target
x = data[["Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]]
y = data["Outcome"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
)

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy: {accuracy}")
print(f"ROC AUC: {roc_auc}")

# Now, load your clinical data which doesn't have an outcome
clinical_data = pd.read_excel("Diabetes (1).xlsx", sheet_name="response")

# Ensure clinical_data has the same features
clinical_features = clinical_data[
    ["Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]
]

# Predict the outcomes for the clinical data
clinical_outcome_pred = rf_model.predict(clinical_features)
clinical_outcome_pred_prob = rf_model.predict_proba(clinical_features)[:, 1]

print(f"clinical_outcome_pred:{clinical_outcome_pred}")
print(f"clinical_outcome_pred_prob:{clinical_outcome_pred_prob}")

# Add predictions to clinical data
clinical_data["PredictedOutcome"] = clinical_outcome_pred
clinical_data["PredictedProbability"] = clinical_outcome_pred_prob

# Save the predictions
clinical_data.to_csv("clinicalData_with_predictions.csv", index=False)
