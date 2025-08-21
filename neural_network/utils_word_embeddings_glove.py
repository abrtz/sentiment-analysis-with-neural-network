import numpy as np
import gensim.downloader as api

model = api.load("glove-wiki-gigaword-200")   ### as seen at https://github.com/piskvorky/gensim-data on 21st Feb 2024

# Map words to embeddings and represent reviews as the average of word embeddings
def get_mean_embedding(review_tokens):
    """
    Map words to embeddings and represent reviews as the average of word embeddings.
    Take a list of tokenized words representing a review and map each word to its corresponding word embedding using a pre-trained word embedding model. 
    If any of the words in the review are not present in the pre-trained word embedding model, their embeddings are skipped, and the review embedding is computed based on the remaining words. 
    If none of the words are present in the model, a zero vector of the same dimensionality as the word embeddings is returned.
    Then compute the average of these word embeddings to obtain a single vector representation for the entire review.
    
    Return numpy array representing the embedding for the entire review. 
    
    Parameters:
    - review_tokens (list of str): a list of tokenized words representing a review.
    """
    
    embeddings = [model[word] for word in review_tokens if word in model]
    if embeddings:
        return np.mean(embeddings, axis=0) # calculating the mean of the word embeddings in the review as a single vector
    else:
        # return a zero vector for out-of-vocabulary words of the size of the model (50 dim)
        return np.zeros(model.vector_size)