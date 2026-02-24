from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier as KNNC
from sklearn.linear_model import LogisticRegression as LogiR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import StackingClassifier
from sklearn import svm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import  accuracy_score
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def load_data(file_path):
    """Load and preprocess the dataset"""
    data = pd.read_csv(file_path,encoding='gbk')

    data.drop("V1", axis=1, inplace=True)
    numeric_columns = data.columns[1:]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)
    data['label'] = data['label'].astype('float32')

    return data

def prepare_data(data):
    """Prepare training and test datasets"""
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

def initialize_classifier():
    clf1 = LogiR(max_iter=3000, C=0.1, random_state=1412, n_jobs=8)
    clf2 = RFC(n_estimators=165, max_features="sqrt", max_depth=4, min_samples_leaf=4, random_state=1412,
               n_jobs=8)
    clf3 = svm.SVC(C=10)
    clf4 = KNNC(n_neighbors=10, n_jobs=8)

    estimators = [("Logistic Regression", clf1), ("RandomForest", clf2), ("svm", clf3), ("KNN", clf4)]

    final_estimator = RFC(n_estimators=100
                          , min_impurity_decrease=0.0025
                          , random_state=420, n_jobs=8)

    clf = StackingClassifier(estimators=estimators
                             , final_estimator=final_estimator
                             ,cv=5
                             , n_jobs=8)
    return estimators,clf

def fusion_estimators(clf,X_train,Y_train,X_test,Y_test):
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    results = cross_validate(clf, X_train, Y_train
                             , cv=cv
                             , scoring="accuracy"
                             , n_jobs=-1
                             , return_train_score=True
                             , verbose=False)
    test = clf.fit(X_train, Y_train).score(X_test, Y_test)
    print("train_score:{}".format(results["train_score"].mean())
          , "\n cv_mean:{}".format(results["test_score"].mean())
          , "\n test_score:{}".format(test)
          )

def individual_estimators(estimators,X_train,Y_train,X_test,Y_test):
    class_names = ['anti-dengue', 'anti-influenza', 'anti-tetanus', 'anti-sars-cov2', 'anti-Tuberculosis']

    for estimator in estimators:
        cv = KFold(n_splits=5, shuffle=True, random_state=1400)
        results = cross_validate(estimator[1], X_train, Y_train, cv=cv,
                                 scoring=("accuracy", "precision_weighted", "recall_weighted", "f1_weighted"),
                                 n_jobs=-1, return_train_score=True, verbose=False)
        estimator[1].fit(X_train, Y_train)
        Y_pred = estimator[1].predict(X_test)

        test_accuracy = accuracy_score(Y_test, Y_pred)
        test_precision = precision_score(Y_test, Y_pred, average='weighted', zero_division=0)
        test_recall = recall_score(Y_test, Y_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(Y_test, Y_pred, average='weighted', zero_division=0)

        print(f'{estimator[0]}: \n'
              f' Test Accuracy: {test_accuracy:.3f}\n'
              f' Test F1: {test_f1:.3f}\n'
              f' Test Precision: {test_precision:.3f}\n'
              f' Test Recall: {test_recall:.3f}\n')

def shap_explainer(data,clf,X_test,):
    explainer = shap.KernelExplainer(clf.predict_proba, X_test)
    shap_values = explainer.shap_values(X_test)
    feature_names = data.columns[1:]
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    plt.show()


if __name__ == "__main__":
    # Load data
    file_path = "data.csv"
    data = load_data(file_path)

    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(data)

    # Initialize Classifier
    estimators,clf = initialize_classifier()

    fusion_estimators(clf,X_train,y_train,X_test,y_test)

    individual_estimators(estimators,X_train,y_train,X_test,y_test)

    shap_explainer(data,clf,X_test)




