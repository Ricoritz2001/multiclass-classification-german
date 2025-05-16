import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_preprocessing import load_and_prepare_data_from_parquet, filter_top_classes, get_stopwords
from src.utils import load_label_encoder
from src.embedding import get_spacy_doc_vectors
from src.train_classical_model import remove_punctuation, clean_for_tfidf  


def evaluate_classical_and_plot(X_test_emb, y_test_le, model_path, le):
    model = joblib.load(model_path)
    preds = model.predict(X_test_emb)
    acc = accuracy_score(y_test_le, preds)
    
    report_str = classification_report(y_test_le, preds, target_names=le.classes_)
    print(f"\nClassification Report for {model_path}:\n{report_str}")
    
    cm = confusion_matrix(y_test_le, preds)
    print(f"\nConfusion Matrix for {model_path}:\n{cm}")
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 16))
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax1)
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    ax1.set_title(f"Confusion Matrix for {model_path}", fontsize=14)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10)    
    
    ax2.axis('off')
    ax2.text(0.01, 0.5, report_str, fontsize=12, family='monospace', va='center')
    ax2.set_title("Classification Report", fontsize=14)
    
    plt.tight_layout()
    output_png = model_path.replace(".joblib", "_combined_report.png")
    plt.savefig(output_png, dpi=300)
    plt.close()
    
    return acc


def main():
    use_tfidf = False  # Set to False to use FastText/spaCy embeddings

    le = load_label_encoder("./models/label_encoder.joblib")
    downsampled_file = "./data/downsampled_df.parquet"
    X_train, X_test, y_train, y_test = load_and_prepare_data_from_parquet(downsampled_file)

    X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered = filter_top_classes(
        X_train, y_train, X_test, y_test, top_n=20
    )

    y_train_le = le.transform(y_train_filtered)
    y_test_le = le.transform(y_test_filtered)

    if use_tfidf:
        print("Using TF-IDF vectorization...")
        stop_words = set(get_stopwords('german'))

        X_train_clean = [clean_for_tfidf(text, stop_words) for text in X_train_filtered]
        X_test_clean = [clean_for_tfidf(text, stop_words) for text in X_test_filtered]

        tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X_train_features = tfidf.fit_transform(X_train_clean)
        X_test_features = tfidf.transform(X_test_clean)

        classical_models = {
            "LogisticRegression": "./models/logistic_regression_model_TFIDF.joblib",
            "RandomForest": "./models/random_forest_model_TFIDF.joblib",
            "XGBoost": "./models/xgboost_model_TFIDF.joblib",
            "KNN": "./models/knn_model_TFIDF.joblib",
            "SVM": "./models/svc_model_TFIDF.joblib"
        }

    else:
        print("Using spaCy (FastText) embeddings...")
        X_train_clean = [remove_punctuation(text) for text in X_train_filtered]
        X_test_clean = [remove_punctuation(text) for text in X_test_filtered]

        X_train_features = get_spacy_doc_vectors(X_train_clean)
        X_test_features = get_spacy_doc_vectors(X_test_clean)

        classical_models = {
            "LogisticRegression": "./models/logistic_regression_model_FastText.joblib",
            "RandomForest": "./models/random_forest_model_FastText.joblib",
            "XGBoost": "./models/xgboost_model_FastText.joblib",
            "KNN": "./models/knn_model_FastText.joblib",
            "SVM": "./models/svc_model_FastText.joblib"
        }

    for name, path in classical_models.items():
        try:
            acc = evaluate_classical_and_plot(X_test_features, y_test_le, path, le)
            print(f"{name} ({'TF-IDF' if use_tfidf else 'FastText'}) Accuracy: {acc}")
        except Exception as e:
            print(f"Error evaluating {name} ({'TF-IDF' if use_tfidf else 'FastText'}): {e}")


if __name__ == "__main__":
    main()
