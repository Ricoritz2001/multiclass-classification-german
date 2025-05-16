import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

# Import your custom data preparation and utility functions
from src.data_preprocessing import load_and_prepare_data_from_parquet, filter_top_classes
from src.utils import load_label_encoder

def evaluate_transformer_and_plot(model_path, raw_test_ds, le):
    """
    Evaluate a transformer model using its own tokenizer and then create a combined plot
    that shows the confusion matrix and classification report in a single PNG.
    
    Parameters:
      model_path (str): Path to the transformer model.
      raw_test_ds (Dataset): Raw Hugging Face dataset with 'text' and 'label' fields.
      le: The label encoder.
      
    Returns:
      Accuracy (float)
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    
    # Tokenize the raw test dataset using this model's tokenizer
    tokenized_test = raw_test_ds.map(tokenize_function, batched=True)
    tokenized_test = tokenized_test.remove_columns(["text"])
    tokenized_test = tokenized_test.rename_column("label", "labels")
    tokenized_test.set_format("torch")
    
    test_dataloader = torch.utils.data.DataLoader(tokenized_test, batch_size=16, shuffle=False)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Ensure all entries are tensors
            for key in batch:
                if not torch.is_tensor(batch[key]):
                    batch[key] = torch.tensor(batch[key])
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            labels = batch["labels"]
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels)
            labels = labels.to(device)
            
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
    # Convert numeric predictions back to string labels using the label encoder
    predicted_labels = le.inverse_transform(all_predictions)
    true_labels = le.inverse_transform(all_labels)
    acc = accuracy_score(true_labels, predicted_labels)
    report_str = classification_report(true_labels, predicted_labels, target_names=le.classes_)
    cm = confusion_matrix(true_labels, predicted_labels)
    
    print(f"\nTransformer Model ({model_path}) Accuracy: {acc:.4f}")
    print(report_str)
    
    # Create a combined figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 16))
    
    # Plot confusion matrix on ax1
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax1)
    ax1.set_xlabel("Predicted Label", fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel("True Label", fontsize=12)
    ax1.set_title(f"Confusion Matrix for {model_path.split('/')[-1]}", fontsize=14)
    
    # Plot the classification report as text on ax2
    ax2.axis('off')  # Hide axis
    ax2.text(0.01, 0.5, report_str, fontsize=12, family='monospace', va='center')
    ax2.set_title("Classification Report", fontsize=14)
    
    plt.tight_layout()
    output_png = model_path.replace(".joblib", "_combined_report.png")
    # In case the model path doesn't contain '.joblib', simply append the suffix:
    if output_png == model_path:
        output_png = model_path + "_combined_report.png"
    plt.savefig(output_png, dpi=300)
    plt.close()
    
    return acc

def main():
    le = load_label_encoder("./models/label_encoder.joblib")
    downsampled_file = "./data/downsampled_df.parquet"
    
    # Load and prepare data 
    X_train, X_test, y_train, y_test = load_and_prepare_data_from_parquet(downsampled_file)
    X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered = filter_top_classes(
        X_train, y_train, X_test, y_test, top_n=20
    )
    
    df_test = {"text": X_test_filtered, "label": y_test_filtered}
    raw_test_ds = Dataset.from_dict(df_test)
    
    # Encode labels in the test dataset
    def encode_labels(examples):
        examples["label"] = le.transform(examples["label"])
        return examples
    raw_test_ds = raw_test_ds.map(encode_labels, batched=True)
    
    # List of finetuned model to evaluate, recommend evaluating one at the time
    transformer_model_paths = [
        #"./models/mBert_finetuned_model",
        "./models/e5_model_finetuned_model"
       # "./models/DistilBert_finetuned_model",
       # "./models/xlm-roberta_finetuned_model"
    ]
    
    for model_path in transformer_model_paths:
        try:
            acc = evaluate_transformer_and_plot(model_path, raw_test_ds, le)
            print(f"{model_path.split('/')[-1]} Accuracy: {acc}")
        except Exception as e:
            print(f"Error evaluating transformer model {model_path}: {e}")

if __name__ == "__main__":
    main()
