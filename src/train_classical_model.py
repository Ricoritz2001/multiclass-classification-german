import time
import re
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Project-specific imports
from src.data_preprocessing import load_and_prepare_data_from_parquet, filter_top_classes, get_stopwords
from src.utils import encode_labels, save_label_encoder
from src.embedding import get_spacy_doc_vectors


USE_TFIDF = True  # Set to True to use TF-IDF, False for FastText/spaCy embeddings

#Text cleaning functions

def remove_punctuation(text):
    return re.sub(r'[^\wäöüÄÖÜß\s]', '', text)

def clean_for_tfidf(text, stopwords_set):
    text = remove_punctuation(text.lower())
    return ' '.join([word for word in text.split() if word not in stopwords_set])


def grid_search_model(model, param_grid, X_train, y_train, model_name):
    grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    start = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"{model_name} grid search took {elapsed: .2f} seconds")
    return grid.best_estimator_


def main():
    # File paths
    downsampled_file = "./data/downsampled_df.parquet"
    label_encoder_path = "./models/label_encoder.joblib"

    # Load and split data
    X_train, X_test, y_train, y_test = load_and_prepare_data_from_parquet(downsampled_file)

    # Filter to top N categories
    X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered = filter_top_classes(
        X_train, y_train, X_test, y_test, top_n=20
    )


    if USE_TFIDF:
        print("Using TF-IDF vectorization...")
        stop_words = set(get_stopwords('german'))

        X_train_clean = [clean_for_tfidf(text, stop_words) for text in X_train_filtered]
        X_test_clean = [clean_for_tfidf(text, stop_words) for text in X_test_filtered]

        tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X_train_vec = tfidf.fit_transform(X_train_clean)
        X_test_vec = tfidf.transform(X_test_clean)

    else:
        print("Using spaCy (FastText) embeddings...")
        X_train_clean = [remove_punctuation(text) for text in X_train_filtered]
        X_test_clean = [remove_punctuation(text) for text in X_test_filtered]

        X_train_vec = get_spacy_doc_vectors(X_train_clean)
        X_test_vec = get_spacy_doc_vectors(X_test_clean)


    y_train_le, y_test_le, le = encode_labels(y_train_filtered, y_test_filtered)
    save_label_encoder(le, label_encoder_path)


    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg_param_grid = {
        'C': [0.1, 1.0, 10.0],
        'solver': ['lbfgs', 'liblinear']
    }
    best_log_reg = grid_search_model(log_reg, log_reg_param_grid, X_train_vec, y_train_le, "Logistic Regression")
    y_pred_lr = best_log_reg.predict(X_test_vec)
    print("LogisticRegression Accuracy:", accuracy_score(y_test_le, y_pred_lr))
    joblib.dump(best_log_reg, f"./models/logistic_regression_model_{'TFIDF' if USE_TFIDF else 'FastText'}.joblib")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_features='sqrt')
    rf_param_grid = {'max_depth': [10, 20]}
    best_rf = grid_search_model(rf, rf_param_grid, X_train_vec, y_train_le, "Random Forest")
    y_pred_rf = best_rf.predict(X_test_vec)
    print("RandomForest Accuracy:", accuracy_score(y_test_le, y_pred_rf))
    joblib.dump(best_rf, f"./models/random_forest_model_{'TFIDF' if USE_TFIDF else 'FastText'}.joblib")

    # XGBoost
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_param_grid = {
        'learning_rate': [0.1, 0.3],
        'max_depth': [3, 6]
    }
    best_xgb = grid_search_model(xgb_clf, xgb_param_grid, X_train_vec, y_train_le, "XGBoost")
    y_pred_xgb = best_xgb.predict(X_test_vec)
    print("XGBoost Accuracy:", accuracy_score(y_test_le, y_pred_xgb))
    joblib.dump(best_xgb, f"./models/xgboost_model_{'TFIDF' if USE_TFIDF else 'FastText'}.joblib")

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn_param_grid = {'n_neighbors': [3, 5, 7]}
    best_knn = grid_search_model(knn, knn_param_grid, X_train_vec, y_train_le, "KNN")
    y_pred_knn = best_knn.predict(X_test_vec)
    print("KNN Accuracy:", accuracy_score(y_test_le, y_pred_knn))
    joblib.dump(best_knn, f"./models/knn_model_{'TFIDF' if USE_TFIDF else 'FastText'}.joblib")

    # SVC
    svc = LinearSVC(random_state=42)
    svc_param_grid = {'C': [0.1, 1.0, 10.0]}
    best_svc = grid_search_model(svc, svc_param_grid, X_train_vec, y_train_le, "LinearSVC")
    y_pred_svc = best_svc.predict(X_test_vec)
    print("LinearSVC Accuracy:", accuracy_score(y_test_le, y_pred_svc))
    joblib.dump(best_svc, f"./models/svc_model_{'TFIDF' if USE_TFIDF else 'FastText'}.joblib")

if __name__ == "__main__":
    main()
