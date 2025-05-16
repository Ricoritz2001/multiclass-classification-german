import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.data_preprocessing import load_and_prepare_data_from_parquet, filter_top_classes

# Load and prepare data from the single downsampled file
downsampled_file = "./data/downsampled_df.parquet"
X_train, X_test, y_train, y_test = load_and_prepare_data_from_parquet(downsampled_file)

# Optionally filter to only the top 20 classes
X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered = filter_top_classes(
    X_train, y_train, X_test, y_test, top_n=20
)

# 
train_counts = pd.Series(y_train_filtered).value_counts().sort_index()
test_counts = pd.Series(y_test_filtered).value_counts().sort_index()

fig, axes = plt.subplots(nrows=2, figsize=(20, 16))  

# Plot training set distribution
sns.barplot(x=train_counts.index, y=train_counts.values, ax=axes[0], color="steelblue")
axes[0].set_title("Training Set Class Distribution", fontsize=16)
axes[0].set_xlabel("Category", fontsize=14)
axes[0].set_ylabel("Frequency", fontsize=14)
axes[0].tick_params(axis='x', labelrotation=45, labelsize=12)

# Plot test set distribution
sns.barplot(x=test_counts.index, y=test_counts.values, ax=axes[1], color="steelblue")
axes[1].set_title("Test Set Class Distribution", fontsize=16)
axes[1].set_xlabel("Category", fontsize=14)
axes[1].set_ylabel("Frequency", fontsize=14)
axes[1].tick_params(axis='x', labelrotation=45, labelsize=12)

plt.tight_layout()
plt.savefig("class_distribution2.png", dpi=300)
