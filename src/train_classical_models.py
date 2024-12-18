import time
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from .data_preprocessing import load_and_prepare_data, filter_top_classes, get_stopwords
from .utils import encode_labels, save_label_encoder

def main():
    # File paths
    categories_file = "./data/Categories_manual_adapted.xlsx"
    data_file = "./data/data.csv"
    new_data_file = "./data/new_data_updated.csv"
    label_encoder_path = "./models/label_encoder.joblib"

    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(categories_file, data_file, new_data_file)
    X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered = filter_top_classes(X_train, y_train, X_test, y_test, top_n=30)

    # TF-IDF
    stop_words = get_stopwords('german')
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words=stop_words)
    X_train_tfidf = tfidf.fit_transform(X_train_filtered)
    X_test_tfidf = tfidf.transform(X_test_filtered)

    # Encode labels
    y_train_le, y_test_le, le = encode_labels(y_train_filtered, y_test_filtered)
    save_label_encoder(le, label_encoder_path)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    start_time = time.time()
    log_reg.fit(X_train_tfidf, y_train_le)
    train_time = time.time() - start_time
    y_pred_lr = log_reg.predict(X_test_tfidf)
    acc_lr = accuracy_score(y_test_le, y_pred_lr)
    print("LogisticRegression Accuracy:", acc_lr)
    joblib.dump(log_reg, "./models/logistic_regression_model.joblib")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_tfidf, y_train_le)
    y_pred_rf = rf.predict(X_test_tfidf)
    acc_rf = accuracy_score(y_test_le, y_pred_rf)
    print("RandomForest Accuracy:", acc_rf)
    joblib.dump(rf, "./models/random_forest_model.joblib")

    # XGBoost
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_clf.fit(X_train_tfidf, y_train_le)
    y_pred_xgb = xgb_clf.predict(X_test_tfidf)
    acc_xgb = accuracy_score(y_test_le, y_pred_xgb)
    print("XGBoost Accuracy:", acc_xgb)
    joblib.dump(xgb_clf, "./models/xgboost_model.joblib")

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_tfidf, y_train_le)
    y_pred_knn = knn.predict(X_test_tfidf)
    acc_knn = accuracy_score(y_test_le, y_pred_knn)
    print("KNN Accuracy:", acc_knn)
    joblib.dump(knn, "./models/knn_model.joblib")

    # SVM
    svm = SVC(kernel="linear", random_state=42)
    svm.fit(X_train_tfidf, y_train_le)
    y_pred_svm = svm.predict(X_test_tfidf)
    acc_svm = accuracy_score(y_test_le, y_pred_svm)
    print("SVM Accuracy:", acc_svm)
    joblib.dump(svm, "./models/svm_model.joblib")


if __name__ == "__main__":
    main()
