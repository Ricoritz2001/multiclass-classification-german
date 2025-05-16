import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from wordcloud import WordCloud
from nltk import ngrams
from collections import Counter
from src.data_preprocessing import get_stopwords


from src.data_preprocessing import load_and_prepare_data_from_parquet, filter_top_classes
from src.utils import load_label_encoder

def generate_wordcloud_for_pair(true_label, pred_label, texts, true_labels, predicted_labels, output_path):
    """
    For a given misclassification pair, filter texts and generate a wordcloud based on bigrams.
    """
    german_stopwords = set(get_stopwords("german"))
    # Filter indices for texts that belong to this misclassified pair
    indices = [i for i, (t, p) in enumerate(zip(true_labels, predicted_labels)) 
               if t == true_label and p == pred_label]
    
    if not indices:
        print(f"No texts found for misclassification pair: {true_label} -> {pred_label}")
        return
    
    misclassified_texts = [texts[i] for i in indices]
    
    # If texts are already preprocessed in 'text_punc', you can skip cleaning
    # Here we simply lowercase them.
    def simple_preprocess(text):
        return text.lower()
    
    preprocessed_texts = [simple_preprocess(t) for t in misclassified_texts]
    
    # Extract bigrams from each text
    bigrams = []
    for text in preprocessed_texts:
        tokens = text.split()
        tokens = [t for t in tokens if t not in german_stopwords]
        bigrams.extend([" ".join(b) for b in ngrams(tokens, 2)])
    
    # Count bigram frequencies
    bigram_freq = Counter(bigrams)
    
    # Generate wordcloud from the bigram frequencies
    wc = WordCloud(width=800, height=400, background_color="white", collocations=False)
    wc.generate_from_frequencies(bigram_freq)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud: {true_label} misclassified as {pred_label}")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"WordCloud saved to {output_path}")

def evaluate_transformer_and_plot(model_path, raw_test_ds, le):
    """
    Evaluate a transformer model, create combined plots (confusion matrix and classification report),
    analyze misclassifications that exceed a threshold, and generate a wordcloud for each misclassified pair.
    """
    # Save original texts 
    original_texts = raw_test_ds["text"]

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    
    tokenized_test = raw_test_ds.map(tokenize_function, batched=True)
    tokenized_test = tokenized_test.remove_columns(["text"])
    tokenized_test = tokenized_test.rename_column("label", "labels")
    tokenized_test.set_format("torch")
    
    test_dataloader = torch.utils.data.DataLoader(tokenized_test, batch_size=16, shuffle=False)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            labels = batch["labels"].to(device)
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    predicted_labels = le.inverse_transform(all_predictions)
    true_labels = le.inverse_transform(all_labels)
    acc = accuracy_score(true_labels, predicted_labels)
    report_str = classification_report(true_labels, predicted_labels, target_names=le.classes_)
    cm = confusion_matrix(true_labels, predicted_labels)
    
    print(f"\nTransformer Model ({model_path}) Accuracy: {acc:.4f}")
    print(report_str)
    
    # Misclassification Analysis
    threshold = 250
    misclassified_pairs = []
    for i, true_label in enumerate(le.classes_):
        for j, pred_label in enumerate(le.classes_):
            if i != j and cm[i, j] >= threshold:
                misclassified_pairs.append((true_label, pred_label, cm[i, j]))
    
    if misclassified_pairs:
        misclassified_df = pd.DataFrame(misclassified_pairs, columns=["True Label", "Predicted Label", "Count"])
        misclassification_table_path = f"{model_path.split('/')[-1]}_misclassified_pairs.csv"
        misclassified_df.to_csv(misclassification_table_path, index=False)
        print("Misclassified pairs (Count >= 250):")
        print(misclassified_df)
        print(f"Misclassification table saved to {misclassification_table_path}")
    else:
        print("No misclassified pairs exceed the threshold.")
    
    #  Generate WordClouds for Each Misclassified Pair
    for true_label, pred_label, count in misclassified_pairs:
        wc_filename = f"{model_path.split('/')[-1]}_{true_label}_to_{pred_label}_wordcloud.png"
        generate_wordcloud_for_pair(true_label, pred_label, original_texts, true_labels, predicted_labels, wc_filename)
    
    # Plot Combined Figure 
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 16))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax1)
    ax1.set_xlabel("Predicted Label", fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel("True Label", fontsize=12)
    ax1.set_title(f"Confusion Matrix for {model_path.split('/')[-1]}", fontsize=14)
    
    ax2.axis('off')
    ax2.text(0.01, 0.5, report_str, fontsize=12, family='monospace', va='center')
    ax2.set_title("Classification Report", fontsize=14)
    
    plt.tight_layout()
    output_png = model_path.replace(".joblib", "_combined_report4.png")
    if output_png == model_path:
        output_png = model_path + "_combined_report3.png"
    plt.savefig(output_png, dpi=300)
    plt.close()
    
    return acc

def main():
    le = load_label_encoder("./models/label_encoder.joblib")
    downsampled_file = "./data/downsampled_df.parquet"
    
    X_train, X_test, y_train, y_test = load_and_prepare_data_from_parquet(downsampled_file)
    X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered = filter_top_classes(
        X_train, y_train, X_test, y_test, top_n=20
    )
    
    df_test = {"text": X_test_filtered, "label": y_test_filtered}
    raw_test_ds = Dataset.from_dict(df_test)
    
    def encode_labels(examples):
        examples["label"] = le.transform(examples["label"])
        return examples
    raw_test_ds = raw_test_ds.map(encode_labels, batched=True)
    
    # Only evaluate the given model ( e.g., e5, xlm, mbert, distilbert)
    model_path = "./models/e5_finetuned_model"
    try:
        acc = evaluate_transformer_and_plot(model_path, raw_test_ds, le)
        print(f"{model_path.split('/')[-1]} Accuracy: {acc}")
    except Exception as e:
        print(f"Error evaluating transformer model {model_path}: {e}")

if __name__ == "__main__":
    main()
