import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

def load_and_prepare_data(categories_file, data_file, new_data_file):
    # Load category mapping
    df2 = pd.read_excel(categories_file)
    df2.columns = ["category", "count", "cat-l2", "cat-l3"]

    # Load initial data
    df1 = pd.read_csv(data_file)
    df1.columns = ["new_number", "news_text", "category"]

    # Map categories to cat-l3
    mapping = df2.set_index('category')['cat-l3'].to_dict()
    df1['cat-l3'] = df1['category'].map(mapping)

    # Drop rows with NaN in cat-l3
    df1 = df1.dropna(subset=['cat-l3'])

    # Load updated data
    df = pd.read_csv(new_data_file)
    df.columns = ["new_number", "news_text", "category", "cat-l3"]
    df = df.dropna(subset=["news_text", "cat-l3"])

    X = df["news_text"].values
    y = df["cat-l3"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test

def filter_top_classes(X_train, y_train, X_test, y_test, top_n=30):
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
