import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_data(data, imputer=None, scaler=None, fit=False):
    features = [
        "Glucose",
        "BloodPressure",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Pregnancies",
    ]
    X = data[features]

    if fit:
        imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=features)
        return X_scaled, imputer, scaler
    else:
        X_imputed = pd.DataFrame(imputer.transform(X), columns=features)
        X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=features)
        return X_scaled


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train):
    rf_model = RandomForestClassifier(random_state=42, criterion="entropy")
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
    return grid_search.best_estimator_


def save_model(model, filepath):
    joblib.dump(model, filepath)


def load_model(filepath):
    return joblib.load(filepath)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_prob),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    return y_pred, y_pred_prob, metrics


def plot_roc_auc(y_test, y_pred_prob, title="ROC AUC Curve"):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.title(title)
    plt.show()


def plot_feature_importance(model, features, title="Feature Importances"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title(title)
    plt.bar(range(len(features)), importances[indices], align="center")
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
    plt.xlim([-1, len(features)])
    plt.show()


def main():
    # Load dataset
    data = load_data("pimaTesting.csv")
    y = data["Outcome"]

    # Preprocess data
    X, imputer, scaler = preprocess_data(data, fit=True)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train model
    best_rf_model = train_model(X_train, y_train)

    # Save the model
    save_model(best_rf_model, "best_rf_model.pkl")

    # Load the model (optional)
    # best_rf_model = load_model('best_rf_model.pkl')

    # Evaluate the model
    y_pred, y_pred_prob, metrics = evaluate_model(best_rf_model, X_test, y_test)
    print(metrics)

    # Plot ROC AUC curve
    plot_roc_auc(y_test, y_pred_prob, title="ROC AUC Curve (Test Data)")

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix (Test Data)")

    # Plot feature importance
    plot_feature_importance(best_rf_model, X.columns)

    # Load and preprocess clinical data
    clinical_data = pd.read_excel("Diabetes (1).xlsx", sheet_name="response")
    clinical_features_scaled = preprocess_data(clinical_data, imputer, scaler)

    # Predict outcomes for clinical data
    clinical_outcome_pred = best_rf_model.predict(clinical_features_scaled)
    clinical_outcome_pred_prob = best_rf_model.predict_proba(clinical_features_scaled)[
        :, 1
    ]
    clinical_data["PredictedOutcome"] = clinical_outcome_pred
    clinical_data["PredictedProbability"] = clinical_outcome_pred_prob

    print(f"clinical_outcome_pred:{clinical_outcome_pred}")
    print(f"clinical_outcome_pred_prob:{clinical_outcome_pred_prob}")

    # Plot ROC AUC curve for clinical data
    plot_roc_auc(
        clinical_data["PredictedOutcome"],
        clinical_data["PredictedProbability"],
        title="ROC AUC Curve (Clinical Data)",
    )

    # Plot confusion matrix for clinical data
    plot_confusion_matrix(
        clinical_data["PredictedOutcome"],
        clinical_data["PredictedOutcome"],
        title="Confusion Matrix (Clinical Data)",
    )

    # Feature importance for clinical data (reuse from training data)
    plot_feature_importance(
        best_rf_model, X.columns, title="Feature Importances (Clinical Data)"
    )


if __name__ == "__main__":
    main()
