# German News Data Classification


This project aims to classify German news articles into predefined categories using both classical machine learning models (Logistic Regression, Random Forest, XGBoost, KNN, SVM) and transformer-based models (e.g. Distilbert, XLM-RoBERTa). The repository provides scripts for data preprocessing, training, and evaluating models, as well as instructions on how to reproduce results.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training the Models](#training-the-models)
  - [Classical Models](#classical-models)
  - [Transformer-based Models](#transformer-based-models)
- [Evaluating the Models](#evaluating-the-models)
- [Inference / Using the Models](#inference--using-the-models)
- [Customizing the Pipeline](#customizing-the-pipeline)

## Overview

The project uses:
- **Classical ML models**: Logistic Regression, Random Forest, XGBoost, KNN, LinearSVC.
- **Transformer-based models**: Fine-tuned German language models from Hugging Face ( `distilbert-base-german-cased`, `xlm-roberta-large`, `bert-base-multilingual-cased`,`intfloat/multilingual-e5-large` ), leveraging the `transformers` and `datasets` libraries.

The primary goal is to predict the `category` (aggregated category) labels for German news articles given their text.

## Project Structure
```bash
German-News-Data-Classification/
├─ data/
│  ├─ downsampled_df.parquet
│  
│  
├─ models/
│  ├─ logistic_regression_model.joblib
│  ├─ random_forest_model.joblib
│  ├─ xgboost_model.joblib
│  ├─ knn_model.joblib
│  ├─ LinearSVC_model.joblib
│  ├─ label_encoder.joblib
│  └─ german_finetuned_model/
│     ├─ config.json
│     ├─ pytorch_model.bin
│     ├─ tokenizer.json
│     ├─ tokenizer_config.json
│     └─ ... (other tokenizer/model files)
├─ src/
│  ├─ data_preprocessing.py
|  ├─ dominantWords.py
|  ├─ embedding.py
|  ├─ evaluate_transformer_logit.py
|  ├─ evaluate_transformer_models.py
|  ├─ evaluate.transformer_wordcloud.py
|  ├─ plot_distribution.py
│  ├─ train_classical_models.py
│  ├─ train_transformer_models.py
│  ├─ evaluate_models.py
│  └─ utils.py
├─ classical_model.sh
├─ dominantwords.sh
├─ evaluate_transformer_logit.sh
├─ evaluate_transformer.models.sh
├─ evaluate_transformer_wordcloud.sh
├─ news_classification_setup.sh
├─ README.md
├─ transformer_model.sh
└─ requirements.txt
```
- **`data/`**: Contains raw input data (Parquet) for training and evaluation.
- **`models/`**: Contains trained model files and label encoder.
- **`src/`**: Source code for data preprocessing, training, and evaluation.
- **`README.md`**: Documentation and instructions.
- **`requirements.txt`**: Python dependencies.

## Prerequisites

- Python 3.7+
- (Optional) A GPU for faster transformer model training and inference.
- Internet connection (if downloading pretrained models from Hugging Face).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/umarmuaz/German-News-Data-Classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd German-News-Data-Classification
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) To utilize GPU acceleration for training transformer models, ensure you have CUDA and PyTorch GPU version installed.
   
## Data Preparation
Place your data files in the **`data`** directory:

- **`downsampled_df.parquet`**


The scripts assume this filename and will attempt to load it directly.

## Training the Models
### Classical Models
Run the following command to train Logistic Regression, Random Forest, XGBoost, KNN, and LinearSVC models on the filtered dataset (top 20 classes):

```bash
python -m src.train_classical_models
```
This will:

- Load and preprocess the data using FastText-based document embeddings and label encoding. Enable TF-IDF in code by changing
```bash
USE_TFIDF = True
```
- Train each classical model( Logistic regression, Random Forest, XGBoost, KNN, LinaerSVC).
- Save the trained models and the label encoder in the models/ directory.
- optional: Adjust hyperparamters to produce better results

### Transformer-based Models
Run the following command to fine-tune a transformer-based model (XLM-RoBERTa, DistilBERT, e5-mutltilingual and mBERT):

```bash
python -m src.train_transformer_models
```
This will:

- Load and preprocess the data.
- Convert the data into the Hugging Face Dataset format.
- Tokenize and prepare inputs for the model.
- Train the transformer-based model using the Trainer API.
- Save the fine-tuned model to the appropriate subdirectory inside the `models/`folder.
### Evaluating the classical Models
This script evaluates the classical model used.


```bash
python -m src.evaluate_models
```
This script will:

- Load the saved classical models (Logistic Regression, Random Forest, XGBoost, KNN, and LinearSVC).
- Load and preprocess the test data.
- Generate document embeddings using FastText-based vectors (via spaCy) enable TF-IDF
```bash
USE_TFIDF = True
```

- Evaluate each model's performance and print classification reports and confusion matrices.

## Inference / Using the Models
After training, you can load models directly for inference. For example, to load and use the transformer model for new text predictions:

```python
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

le = joblib.load("./models/label_encoder.joblib")
model_path = "./models/german_finetuned_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

texts = ["Dein Beispieltext hier"]
inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

predictions = torch.argmax(logits, dim=1).numpy()
predicted_labels = le.inverse_transform(predictions)
print(predicted_labels)
```
## Customizing the Pipeline
- Changing the number of classes: Update **`top_n=30`** in **`train_classical_models.py`** or **`train_transformer_models.py`**.
- Modifying the transformer model: Change **`model_name`** in **`train_transformer_models.py`** to a different Hugging Face model.
- Adjusting hyperparameters: Modify TF-IDF parameters, model hyperparameters, or training arguments in the respective scripts.
