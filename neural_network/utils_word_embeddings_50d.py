import numpy as np

def read_word_embbedding(file_name):
    ### code as seen in https://www.kaggle.com/code/adepvenugopal/nlp-word-embedding-using-glove-vector on Jan 30th
    """Read a GloVe txt file containing pre-trained word embeddings with 6B tokens and 50 dimensions.
    Return the Python set of unique words and the word-to-vector Python dictionary.

    Parameters:
    -'file_name': the path to the txt file storing the GloVe pre-trained embeddings."""
    
    with open(file_name,'r') as f:
        word2vector = {}
        for line in f:
            line = line.strip() #removing white spaces
            words_vec = line.split() #spliting the line into a list of word and vectors.
            word2vector[words_vec[0]] = np.array(words_vec[1:],dtype=float) #making a np array from the vector components of words_vec which is the value of the word in the word2vector dictionary
    return word2vector


#reading and loading the word embedding vector with 50 dimensions
model = read_word_embbedding("./models/glove.6B.50d.txt")


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
        return np.zeros(50)