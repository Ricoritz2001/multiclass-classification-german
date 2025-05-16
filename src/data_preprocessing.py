import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

def load_data_from_parquet(file_path):
    df = pd.read_parquet(file_path)
    df = df.dropna(subset=["text_punc", "category"])
    
    class_counts = df["category"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df["category"].isin(valid_classes)]
    
    X = df["text_punc"].values
    y = df["category"].values
    return X, y
    

def load_and_prepare_data_from_parquet(downsampled_file):
    """
    Loads data from a single Parquet file that contains:
      - news_ID, text_punc, category, and item_date_published
    
    Drops rows with missing 'text_punc' or 'category', filters out classes with fewer than 2 samples,
    and splits the data into training and test sets.
    """
    # Load the downsampled DataFrame
    df = pd.read_parquet(downsampled_file)
    
    # Drop rows with missing text or category information
    df = df.dropna(subset=["text_punc", "category"])
    
    # Filter out classes with fewer than 2 instances
    class_counts = df["category"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df["category"].isin(valid_classes)]
    
    # Extract features and labels using text_punc as the text and category as the label
    X = df["text_punc"].values
    y = df["category"].values
    
    # Split into training and test sets with stratification on the labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def filter_top_classes(X_train, y_train, X_test, y_test, top_n=20):
    """
    Filters the training and test sets to include only the top N classes by frequency in the training set.
    """
    from collections import Counter
    class_counts = Counter(y_train)
    top_classes = [cls for cls, count in class_counts.most_common(top_n)]
    
    train_mask = np.array([label in top_classes for label in y_train])
    test_mask = np.array([label in top_classes for label in y_test])
    
    X_train_filtered = X_train[train_mask]
    y_train_filtered = y_train[train_mask]
    X_test_filtered = X_test[test_mask]
    y_test_filtered = y_test[test_mask]
    
    return X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered

def get_stopwords(language='german'):
    import nltk
    nltk.download('stopwords', quiet=True)
    return stopwords.words(language)
