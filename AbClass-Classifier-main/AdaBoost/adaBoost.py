import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score


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

# Train AdaBoost model
def train_model(X_train_scaled,y_train):
    # Define weak classifier
    weak_classifier = DecisionTreeClassifier(max_depth=6, random_state=42)

    # Create AdaBoost classifier instance
    adaboost = AdaBoostClassifier(base_estimator=weak_classifier, n_estimators=140, random_state=42)

    # Train model
    adaboost.fit(X_train_scaled, y_train)

    # Calculate training set score
    train_score = adaboost.score(X_train_scaled, y_train)
    print(f"Training set score: {train_score}")

    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(adaboost, X_train_scaled, y_train, cv=5)

    # Print cross-validation scores
    print("Cross-validation scores:")
    for i, score in enumerate(cv_scores):
        print(f"Fold {i + 1}: {score}")

    return adaboost

# Evaluate model performance
def evaluate_model(adaboost, X_test_scaled, y_test):
    # Calculate test set score
    test_score = adaboost.score(X_test_scaled, y_test)
    print(f"Test set score: {test_score}")

    # Define scorers
    scoring = {
        'accuracy': 'accuracy',
        'f1_score': make_scorer(f1_score, average='weighted'),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted')
    }

    # Perform cross-validation with multiple metrics
    cv_results_test = cross_validate(adaboost, X_test_scaled, y_test, cv=5, scoring=scoring)

    # Print average cross-validation scores
    print("Test set cross-validation average scores:")
    print(f"Average Accuracy: {cv_results_test['test_accuracy'].mean()}")
    print(f"Average F1 Score: {cv_results_test['test_f1_score'].mean()}")
    print(f"Average Precision: {cv_results_test['test_precision'].mean()}")
    print(f"Average Recall: {cv_results_test['test_recall'].mean()}")

if __name__ == '__main__':
    # Load data
    file_path = "data.csv"
    data = load_data(file_path)

    # Prepare data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(data)

    # Train model
    adaboost = train_model(X_train_scaled, y_train)

    # Evaluate model
    evaluate_model(adaboost, X_test_scaled, y_test)






