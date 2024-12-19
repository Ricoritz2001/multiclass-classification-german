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
- **Classical ML models**: Logistic Regression, Random Forest, XGBoost, KNN, SVM.
- **Transformer-based models**: Fine-tuned German language models from Hugging Face (e.g., `distilbert-base-german-cased`, `xlm-roberta-base`), leveraging the `transformers` and `datasets` libraries.

The primary goal is to predict `category` (category aggregated level 3) labels for German news articles given their text.

## Project Structure
```bash
German-News-Data-Classification/
├─ data/
│  ├─ Categories_manual_adapted.xlsx
│  ├─ data.csv
│  └─ new_data_updated.csv
├─ models/
│  ├─ logistic_regression_model.joblib
│  ├─ random_forest_model.joblib
│  ├─ xgboost_model.joblib
│  ├─ knn_model.joblib
│  ├─ svm_model.joblib
│  ├─ label_encoder.joblib
│  └─ german_finetuned_model/
│     ├─ config.json
│     ├─ pytorch_model.bin
│     ├─ tokenizer.json
│     ├─ tokenizer_config.json
│     └─ ... (other tokenizer/model files)
├─ src/
│  ├─ data_preprocessing.py
│  ├─ train_classical_models.py
│  ├─ train_transformer_models.py
│  ├─ evaluate_models.py
│  └─ utils.py
├─ README.md
└─ requirements.txt
```
- **`data/`**: Contains raw input data (CSV, Excel) for training and evaluation.
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
3. NInstall the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) To utilize GPU acceleration for training transformer models, ensure you have CUDA and PyTorch GPU version installed.
   
## Data Preparation
Place your data files in the **`data`** directory:

- **`Categories_manual_adapted.xlsx`**
- **`data.csv`**
- **`new_data_updated.csv`**

The scripts assume these filenames and will attempt to load them directly.

## Training the Models
### Classical Models
Run the following command to train Logistic Regression, Random Forest, XGBoost, KNN, and SVM models on the filtered dataset (top 30 classes):

```bash
python -m src.train_classical_models
```
This will:

- Load and preprocess the data (applying TF-IDF vectorization and label encoding).
- Train each classical model.
- Save the trained models and the label encoder in the models/ directory.

### Transformer-based Models
Run the following command to fine-tune a transformer-based model (e.g., XLM-RoBERTa):

```bash
python -m src.train_transformer_models
```
This will:

- Load and preprocess the data.
- Convert the data into the Hugging Face Dataset format.
- Tokenize and prepare inputs for the model.
- Train the transformer-based model using the Trainer API.
- Save the fine-tuned model into models/german_finetuned_model/.
### Evaluating the Models
To evaluate both classical and transformer-based models on the test set, run:

```bash
python -m src.evaluate_models
```
This script will:

- Load the saved models.
- Vectorize the test data (for classical models).
- Evaluate each model's accuracy and print a classification report.
- Evaluate the transformer model on the test dataset and print results.

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

## Testing Using colab
We have added the file **`test_using_colab.ipynb`**. Which can be uploaded to the colab. Either train the models from scratch or otherwise just use the pretrain models to test and predict the categories.