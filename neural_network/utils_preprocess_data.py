import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def get_wordnet_pos(tag):
    """Take a POS tag and return the corresponding WordNet POS tag.

    Parameter:
    -tag: a POS tag.
    """
    
    if tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('J'):
        return wordnet.ADJ
    else:
        return wordnet.NOUN  # Default to noun if the POS is not recognized

def preprocess_text(text,remove_digits=True):
    """Read a csv file and clean the text by removing html string using the BeautifulSoup library.
    Apply tokenization and lemmatization.
    Return the df with the preprocessed text in the respective column.
    
    Parameters:
    -reviews_file: path to a csv file containing the final reviews, provided as a Python string.
    -text_column: name of the column in the csv file where the reviews are stored, provided as a Python string.
    """
    
    #removing HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()

    #removing square brackets
    text = re.sub('\[[^]]*\]', '', text)
    
    #removing special characters
    if remove_digits:
        text = re.sub('[^a-zA-Z\s]', '', text)
    else:
        text = re.sub('[^a-zA-Z0-9\s]', '', text)
    
    #lowercasing the text
    text = text.lower()

    #tokenization with NLTK
    tokens = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    #lemmatization with POS tagging
    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(tokens) #applying the NLTK POS tags to tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(pos_tag)) for token, pos_tag in pos_tags] #lemmatizing each token based on its POS
    lemmatized_text = ' '.join(lemmatized_tokens)  #joining lemmatized tokens back into a single string

    return lemmatized_text

###part of the code as seen on https://www.kaggle.com/code/aashidutt3/sentiment-analysis-sephora-reviews/notebook on Jan 23rd

def preprocess_and_read_csv(csv_file_path, text_column='text'):
    """
    Read a csv file into a pandas DataFrame and apply the preprocessing function to the specified text column.
    Return a pandas df with the csv data and the preprocessed text column.
    
    Parameters:
    - csv_file_path (str): path to the CSV file to be read.
    - text_column (str): name of the text column to which the preprocessing function will be applied, default='text' 
    """

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Apply the preprocessing function to the text column
    df['preprocessed_text'] = df[text_column].apply(preprocess_text)

    #after preprocessing it was discovered that there is data in foreign language as Russian, so they are not preprocessed, dropping anything that is not preprocessed
    df.dropna(inplace=True) 
    df = df.reset_index(drop = True)

    return df