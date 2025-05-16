import spacy
import numpy as np

def get_spacy_doc_vectors(texts, model_name="de_core_news_lg"):
    """
    Computes document-level embeddings for a list of texts using a spaCy pretrained model.
    
    Parameters:
      texts (list or array-like): List of text documents.
      model_name (str): Name of the spaCy model to load (make sure the model is installed).
      
    Returns:
      np.array: Array of document vectors.
    """
    # Load the spaCy model 
    nlp = spacy.load(model_name)
    
    # Compute the document vector for each text
    vectors = [nlp(text).vector for text in texts]
    return np.array(vectors)
