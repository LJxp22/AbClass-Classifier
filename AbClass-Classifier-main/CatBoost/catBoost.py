import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Load and preprocess data
def load_data(file_path):
    """Load and preprocess dataset"""
    data = pd.read_csv(file_path, encoding='gbk')
    data.drop("V1", axis=1, inplace=True)

    # Convert numeric columns and handle missing values
    numeric_columns = data.columns[1:]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)

    # Adjust labels to start from 0
    data['label'] = data['label'].astype('float32') - 1

    return data


# Prepare training and test data
def prepare_data(data):
    """Split data into training and test sets"""
    X = data.drop('label', axis=1).values
    y = data['label'].values

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# Train CatBoost model
def train_model(X_train, y_train):
    """Train CatBoost classifier"""
    model = CatBoostClassifier(
        iterations=1000,
        depth=5,
        learning_rate=0.1,
        loss_function='MultiClass',
        verbose=False
    )
    model.fit(X_train, y_train)
    return model


# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    """Evaluate model and print metrics"""
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\nMetrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

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



