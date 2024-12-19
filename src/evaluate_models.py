import joblib
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .data_preprocessing import load_and_prepare_data, filter_top_classes
from .utils import load_label_encoder

def evaluate_classical(X_test_tfidf, y_test_le, model_path, le):
    model = joblib.load(model_path)
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test_le, preds)
    return acc

def evaluate_transformer(test_tokenized, model_path, le):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = torch.utils.data.DataLoader(test_tokenized, batch_size=16, shuffle=False)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            labels = batch["labels"].to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    predicted_labels = le.inverse_transform(all_predictions)
    true_labels = le.inverse_transform(all_labels)
    acc = accuracy_score(true_labels, predicted_labels)
    print("Transformer Model Accuracy:", acc)
    print(classification_report(true_labels, predicted_labels))

def main():
    # Load label encoder
    le = load_label_encoder("./models/label_encoder.joblib")

    # Data paths
    categories_file = "./data/Categories_manual_adapted.xlsx"
    data_file = "./data/data.csv"
    new_data_file = "./data/new_data_updated.csv"

    X_train, X_test, y_train, y_test = load_and_prepare_data(categories_file, data_file, new_data_file)
    X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered = filter_top_classes(X_train, y_train, X_test, y_test, top_n=30)

    # Load TF-IDF and label encoder that was used during training classical models
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.corpus import stopwords
    import nltk
    nltk.download('stopwords', quiet=True)
    stop_words = stopwords.words('german')
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words=stop_words)
    X_train_tfidf = tfidf.fit_transform(X_train_filtered)
    X_test_tfidf = tfidf.transform(X_test_filtered)
    y_train_le = le.transform(y_train_filtered)
    y_test_le = le.transform(y_test_filtered)

    # Evaluate classical models
    classical_models = {
        "LogisticRegression": "./models/logistic_regression_model.joblib",
        "RandomForest": "./models/random_forest_model.joblib",
        "XGBoost": "./models/xgboost_model.joblib",
        "KNN": "./models/knn_model.joblib",
        "SVM": "./models/svm_model.joblib"
    }

    for name, path in classical_models.items():
        acc = evaluate_classical(X_test_tfidf, y_test_le, path, le)
        print(f"{name} Accuracy: {acc}")

    # Evaluate Transformer model
    from datasets import Dataset
    df_test = {"text": X_test_filtered, "label": y_test_filtered}
    test_ds = Dataset.from_dict(df_test)

    def encode_labels(examples):
        examples["label"] = le.transform(examples["label"])
        return examples

    test_ds = test_ds.map(encode_labels, batched=True)
    model_path = "./models/german_finetuned_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

    test_tokenized = test_ds.map(tokenize_function, batched=True)
    test_tokenized = test_tokenized.remove_columns(["text"])
    test_tokenized = test_tokenized.rename_column("label", "labels")
    test_tokenized.set_format("torch")

    evaluate_transformer(test_tokenized, model_path, le)

if __name__ == "__main__":
    main()
