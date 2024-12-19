import pandas as pd
import numpy as np
import joblib
import evaluate
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from .data_preprocessing import load_and_prepare_data, filter_top_classes
from .utils import load_label_encoder

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=preds, references=labels)

def main():
    # Load label encoder
    le = load_label_encoder("./models/label_encoder.joblib")

    # Data paths
    categories_file = "./data/Categories_manual_adapted.xlsx"
    data_file = "./data/data.csv"
    new_data_file = "./data/new_data_updated.csv"

    X_train, X_test, y_train, y_test = load_and_prepare_data(categories_file, data_file, new_data_file)
    X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered = filter_top_classes(X_train, y_train, X_test, y_test, top_n=30)

    # Create Hugging Face datasets
    df_train = pd.DataFrame({"text": X_train_filtered, "label": y_train_filtered})
    df_test = pd.DataFrame({"text": X_test_filtered, "label": y_test_filtered})

    train_ds = Dataset.from_pandas(df_train)
    test_ds = Dataset.from_pandas(df_test)

    # Encode labels
    def encode_labels(examples):
        examples["label"] = le.transform(examples["label"])
        return examples

    train_ds = train_ds.map(encode_labels, batched=True)
    test_ds = test_ds.map(encode_labels, batched=True)

    # Choose a model name 
    # - xlm-roberta-base
    # - distilbert-base-german-cased
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

    train_tokenized = train_ds.map(tokenize_function, batched=True)
    test_tokenized = test_ds.map(tokenize_function, batched=True)

    train_tokenized = train_tokenized.remove_columns(["text"])
    test_tokenized = test_tokenized.remove_columns(["text"])

    train_tokenized = train_tokenized.rename_column("label", "labels")
    test_tokenized = test_tokenized.rename_column("label", "labels")

    train_tokenized.set_format("torch")
    test_tokenized.set_format("torch")

    num_labels = len(le.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir="./models/german_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.evaluate(test_tokenized)
    print("Transformer model evaluation:", results)
    trainer.save_model("./models/german_finetuned_model")

if __name__ == "__main__":
    main()
