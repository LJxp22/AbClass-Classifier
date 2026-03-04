import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

RANDOM_STATE = 42
N_SPLITS = 5

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path, encoding="gbk")

    if "V1" in data.columns:
        data = data.drop(columns=["V1"])

    numeric_columns = data.columns[1:]
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    data = data.dropna()
    data["label"] = data["label"].astype("float32") - 1

    return data

# Train-test split
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

# Build Pipeline
def build_model():
    weak_classifier = DecisionTreeClassifier(
        max_depth=6,
        random_state=RANDOM_STATE,
    )

    adaboost = AdaBoostClassifier(
        estimator=weak_classifier,
        n_estimators=140,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", adaboost),
    ])

    return pipeline
    
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
    )

    print("\n===== 5-Fold Cross Validation (Training Set) =====")
    print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
    print(f"F1: {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
    print(f"Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
    print(f"Recall: {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")

# Test Set Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\n===== Test Set Evaluation =====")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Main
if __name__ == "__main__":
    file_path = "data.csv"
    data = load_data(file_path)

    X_train, X_test, y_train, y_test = split_data(data)

    model = build_model()

    # 5-fold CV on training data
    cross_validate_model(model, X_train, y_train)

    # Train final model on full training data
    model.fit(X_train, y_train)

    # Evaluate on test set
    evaluate_model(model, X_test, y_test)
