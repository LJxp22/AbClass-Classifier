import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier

RANDOM_STATE = 42
N_SPLITS = 5

# Load Data
def load_data(file_path):
    data = pd.read_csv(file_path)

    if "V1" in data.columns:
        data = data.drop(columns=["V1"])

    numeric_columns = data.columns[1:]
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    data = data.dropna()
    data["label"] = data["label"].astype("int") - 1

    return data

# Train/Test Split
def split_data(data):
    X = data.drop("label", axis=1).values
    y = data["label"].values

    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

# Build Model
def build_model(num_classes):
    return XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=num_classes,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        n_jobs=-1,
    )

# Cross Validation
def cross_validate_model(model, X_train, y_train):
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "f1": "f1_weighted",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
    }

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
    )

    print("\n===== 5-Fold Cross Validation (Training Set) =====")
    print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
    print(f"F1: {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
    print(f"Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
    print(f"Recall: {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")

# Test Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\n===== Test Set Evaluation =====")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

# Main
if __name__ == "__main__":
    data = load_data("data.csv")

    X_train, X_test, y_train, y_test = split_data(data)

    num_classes = len(np.unique(y_train))
    model = build_model(num_classes)

    # 5-fold CV on training data
    cross_validate_model(model, X_train, y_train)

    # Train final model
    model.fit(X_train, y_train)

    # Evaluate on test set
    evaluate_model(model, X_test, y_test)
