import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load and preprocess the dataset"""
    data = pd.read_csv(file_path, encoding='gbk')

    data.drop("V1", axis=1, inplace=True)

    # Process numeric columns and missing values
    numeric_columns = data.columns[1:]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)

    # Adjust labels to start from 0
    data['label'] = data['label'].astype('float32') - 1

    return data


def prepare_data(data):
    """Prepare training and test datasets"""
    X = data.drop('label', axis=1).values
    y = data['label'].values

    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train):
    """Train RandomForestClassifier with class balancing"""
    model = RandomForestClassifier(
        n_estimators=165,  # Number of decision trees
        max_depth=5,  # Maximum tree depth
        min_samples_leaf=5,  # Minimum samples per leaf node
        oob_score=True,  # Use out-of-bag samples for evaluation
        random_state=42,  # Ensure reproducibility
        n_jobs=-1,  # Use all CPU cores
        class_weight='balanced'  # Balance class weights
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model with weighted metrics and visualizations"""
    class_names = ['anti-dengue', 'anti-influenza', 'anti-tetanus', 'anti-sars-cov2', 'anti-Tuberculosis', 'other']
    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(y_test, y_pred)

    # Calculate weighted evaluation metrics
    weighted_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    cm_display.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix (Weighted Metrics)')
    plt.tight_layout()
    plt.show()

    # Generate weighted classification report
    print("\nWeighted Classification Report:")
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        average='weighted',
        zero_division=0
    )

    print(f"Test Accuracy: {overall_accuracy:.4f}")
    print(f"Test Out-of-Bag Accuracy: {model.oob_score_:.4f}")
    print(f"Test Weighted Precision: {weighted_precision:.4f}")
    print(f"Test Weighted Recall: {weighted_recall:.4f}")
    print(f"Test Weighted F1-Score: {weighted_f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print(report)

if __name__ == "__main__":
    # Load data
    file_path = "data.csv"
    data = load_data(file_path)

    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(data)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)
