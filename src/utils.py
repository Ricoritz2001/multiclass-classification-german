import os
import joblib
from sklearn.preprocessing import LabelEncoder

def encode_labels(y_train, y_test):
    le = LabelEncoder()
    y_train_le = le.fit_transform(y_train)
    y_test_le = le.transform(y_test)
    return y_train_le, y_test_le, le

def save_label_encoder(le, path):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(le, path)

def load_label_encoder(path):
    return joblib.load(path)
