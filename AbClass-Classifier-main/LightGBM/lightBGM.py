import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
from lightgbm import LGBMClassifier

# Load and preprocess data
def load_data(file_path):
    """Load and preprocess dataset"""
    data = pd.read_csv(file_path)
    data.drop("V1", axis=1, inplace=True)

    # Process numeric columns and handle missing values
    numeric_columns = data.columns[1:]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)

    # Encode label column
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'].astype(str))

    return data

# Prepare training and test data
def prepare_data(data):
    """Split data into training and test sets"""
    X = data.drop('label', axis=1).values
    y = data['label'].values

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train , y_test, scaler

# Train LightBGM model
def train_model(X_train_scaled, y_train):
    # Define LightGBM model
    lgbm_model = LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=500,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=0,
        objective='binary',
        random_state=42,
        n_jobs=-1,
        importance_type='split'
    )

    # Perform 5-fold cross-validation (using accuracy)
    cv_scores = cross_val_score(lgbm_model, X_train_scaled, y_train, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation accuracy:", np.mean(cv_scores))

    # Train the model
    lgbm_model.fit(X_train_scaled, y_train)

    y_train_pred = lgbm_model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("\nTraining Set Evaluation:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(classification_report(y_train, y_train_pred, average='weighted'))

    return lgbm_model

# Evaluate model performance
def evaluate_model(lgbm_model, X_test_scaled, y_test):
    # Evaluate on test set
    y_test_pred = lgbm_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    print("\nTest Set Evaluation:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Weighted Precision: {test_precision:.4f}")
    print(f"Test Weighted Recall: {test_recall:.4f}")
    print(f"Test Weighted F1-Score: {test_f1:.4f}")
    print("Test Classification Report (Weighted):")
    print(classification_report(y_test, y_test_pred, average='weighted'))

if __name__ =='__main__':
    # Load data
    file_path = "data.csv"
    data = load_data(file_path)

    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(data)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

