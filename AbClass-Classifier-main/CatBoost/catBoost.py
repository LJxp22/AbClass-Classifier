import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
N_SPLITS = 5


def load_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess dataset."""
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


def split_data(data: pd.DataFrame):
    """Split dataset into train and test sets."""
    X = data.drop("label", axis=1).values
    y = data["label"].values

    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def build_model() -> CatBoostClassifier:
    """Create CatBoost model."""
    return CatBoostClassifier(
        iterations=1000,
        depth=5,
        learning_rate=0.1,
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        verbose=False,
    )


def cross_validate(model, X_train, y_train):
    """Perform 5-fold stratified cross-validation."""
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    acc_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=skf,
        scoring="accuracy",
    )

    f1_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=skf,
        scoring="f1_weighted",
    )

    print("\n===== 5-Fold Cross Validation =====")
    print(f"Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"Weighted F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)

    print("\n===== Test Set Evaluation =====")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nMetrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")


def main():
    data = load_data("data.csv")
    X_train, X_test, y_train, y_test = split_data(data)

    model = build_model()

    # 5-fold CV (only on training set)
    cross_validate(model, X_train, y_train)

    # Train final model on full training data
    model.fit(X_train, y_train)

    # Evaluate on test set
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
