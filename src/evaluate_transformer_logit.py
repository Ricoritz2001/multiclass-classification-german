import os
import ast
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

from src.data_preprocessing import load_and_prepare_data_from_parquet, filter_top_classes
from src.utils import load_label_encoder

def get_second_highest_index(logits):
    logits_array = np.array(logits)
    sorted_indices = np.argsort(logits_array)
    return sorted_indices[-2]

def round_logits(logit_list, decimals=2):
    return [round(x, decimals) for x in logit_list]

def compute_logit_margin(logits):
    """Compute margin between top 2 logits."""
    top2 = sorted(logits, reverse=True)[:2]
    return round(top2[0] - top2[1], 2) if len(top2) >= 2 else None

def evaluate_and_store_predictions(model_path, raw_test_ds, le):
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

    results = []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
            max_confidences, _ = torch.max(probs, dim=1)

            for i in range(logits.size(0)):
                full_logits = logits[i].cpu().tolist()
                rounded_logits = round_logits(full_logits)
                second_idx = get_second_highest_index(rounded_logits)

                results.append({
                    "true_label": labels[i].cpu().item(),
                    "logits": rounded_logits,
                    "predicted_label": predictions[i].cpu().item(),
                    "confidence": round(max_confidences[i].cpu().item(), 2),
                    "second_highest_index": int(second_idx)
                })

    for res in results:
        res["true_label_name"] = le.classes_[res["true_label"]]
        res["predicted_label_name"] = le.classes_[res["predicted_label"]]
        res["second_highest_label_name"] = le.classes_[res["second_highest_index"]]
        c = res["confidence"]
        if c >= 0.9:
            res["certainty"] = "Confident"
        elif 0.5 <= c < 0.9:
            res["certainty"] = "Uncertain"
        else:
            res["certainty"] = "Very Uncertain"
        res["logit_margin"] = compute_logit_margin(res["logits"])

    df_results = pd.DataFrame(results)
    df_display = df_results[[
        "true_label_name", "logits", "predicted_label_name",
        "second_highest_index", "second_highest_label_name",
        "confidence", "certainty", "logit_margin"
    ]]

    output_csv = model_path.split('/')[-1] + "_predictions_with_logits.csv"
    df_display.to_csv(output_csv, index=False)
    print(f"Predictions with logits and logit margin saved to {output_csv}")

    return df_display

def main():
    le = load_label_encoder("./models/label_encoder.joblib")
    data_path = "./data/downsampled_df.parquet"

    X_train, X_test, y_train, y_test = load_and_prepare_data_from_parquet(data_path)
    X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered = filter_top_classes(
        X_train, y_train, X_test, y_test, top_n=20
    )

    df_test = {"text": X_test_filtered, "label": y_test_filtered}
    raw_test_ds = Dataset.from_dict(df_test)

    def encode_labels(examples):
        examples["label"] = le.transform(examples["label"])
        return examples
    raw_test_ds = raw_test_ds.map(encode_labels, batched=True)

    model_path = "./models/e5_finetuned_model"
    evaluate_and_store_predictions(model_path, raw_test_ds, le)

if __name__ == "__main__":
    main()
