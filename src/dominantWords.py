import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords

# Download German stopwords 
nltk.download('stopwords', quiet=True)
german_stopwords = set(stopwords.words('german'))

# Loads Parquet file 
df = pd.read_parquet("./data/downsampled_df.parquet")

# Create output folder
output_folder = "cat_dominantwords"
os.makedirs(output_folder, exist_ok=True)

# For each category, generate a wordcloud and save it to the output folder
categories = df["category"].unique()
for cat in categories:
    # Concatenate all text in this category
    cat_text = " ".join(df[df["category"] == cat]["text_punc"].tolist())
    
    # Generate wordcloud with German stopwords removed
    wc = WordCloud(width=800, height=400, background_color="white", 
                   collocations=False, stopwords=german_stopwords).generate(cat_text)
    
    # Plot the wordcloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Dominant Words in Category: {cat}")
    
    # Define output filename inside the output folder
    output_filename = os.path.join(output_folder, f"wordcloud_{cat.replace(' ', '_')}.png")
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"WordCloud for category '{cat}' saved to {output_filename}")
