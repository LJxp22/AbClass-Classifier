import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_pipeline() -> Pipeline:
    """Create ML pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=165,
            max_depth=5,
            min_samples_leaf=5,
            oob_score=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])


def cross_validate(model: Pipeline, X_train, y_train):
    """Perform 5-fold cross-validation."""
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    acc_scores = cross_val_score(
        model, X_train, y_train,
        cv=skf,
        scoring="accuracy"
    )

    f1_scores = cross_val_score(
        model, X_train, y_train,
        cv=skf,
        scoring="f1_weighted"
    )

    print("\n===== 5-Fold Cross Validation =====")
    print(f"Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"Weighted F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")


def evaluate(model: Pipeline, X_test, y_test):
    """Evaluate model on test set."""
    class_names = [
        "anti-dengue",
        "anti-influenza",
        "anti-tetanus",
        "anti-sars-cov2",
        "anti-tuberculosis",
        "other",
    ]

    y_pred = model.predict(X_test)

    print("\n===== Test Set Evaluation =====")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Weighted Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Weighted Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def main():
    data = load_data("../data/data.csv")
    X_train, X_test, y_train, y_test = split_data(data)

    model = build_pipeline()

    cross_validate(model, X_train, y_train)

    model.fit(X_train, y_train)

    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
