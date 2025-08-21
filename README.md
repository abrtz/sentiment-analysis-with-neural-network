# Sentiment Analysis with a Neural Network

This repository contains the notebooks to build my own neural network to perform an NLP task. The task chosen is sentiment analysis.

The neural network was built following the steps described in the book Make Your Own Neural Network by Tariq Rashid (2016).

## This repository contains the following files and directories:
- /book_nn_code
	- 01_nn-from-book.ipynb
	- 02_mnist-exploration.ipynb
- /neural_network
	- /models
	- /sephora-data
	- data-exploration.ipynb
	- data-preparation.ipynb
	- 01_sa_bow_unigrams.ipynb
	- 02_sa_bow_bigrams.ipynb
	- 03_sa_bow_trigrams.ipynb
	- 04_sa_bow_tfidf.ipynb
	- 05_sa_word-embeddings-50d.ipynb
	- 06_sa_word-embeddings_200d.ipynb
	- 07_sa_word-embeddings.ipynb
	- nn.py
	- train_test.py
	- utils_data_exploration.oy
	- utils_preprocess_data.py
	- utils_word_embeddings.py
	- utils_word_embeddings_50d.py
	- utils_word_embeddings_glove.py


## Requirements

To run these experiments, make sure libraries required are installed. They are listed in the `requirements.txt` file.

They can be installed with the following command:

pip install -r requirements.txt


### /book_nn_code

## Neural Network from Rashid (2016)


Notebooks 01 and 02 contain the code applied following the code proposed in the abovementioned book to create a NN for image recognition.


### /neural_network

## Data

The complete dataset can be downloaded from: `https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews?resource=download` \
To run the data exploration and preparation, the complete dataset has to be downloaded and placed under the `/sephora-data` directory \

To run the neural network experiments, the preprocessed files are provided under the `/neural_network/sephora-data` directory.

## Models
The word2vec file can be downloaded from: `https://www.kaggle.com/datasets/watts2/glove6b50dtxt` \
The Google news model can be downloaded from: `https://github.com/mmihaltz/word2vec-GoogleNews-vectors` \
Place both models file under the `/models` directory.


To run these experiments, the trained models are provided under the `/neural_network/models` directory together with the vectorizers applied.

## Data exploration and preprocessing 

The notebooks `data-exploration.ipynb` and `data-preparation.ipynb` explore the original data files and prepare the dataset for the neural network experiments to perform the sentiment analysis task. \

## Experiments

- `01_sa_bow_unigrams.ipynb`: training and testing a neural network for sentiment analysis with a unigram BoW

- `02_sa_bow_bigrams.ipynb`: training and testing a neural network for sentiment analysis with a bigram BoW

- `03_sa_bow_trigrams.ipynb`: training and testing a neural network for sentiment analysis with a trigram BoW

- `04_sa_bow_tfidf.ipynb`: training and testing a neural network for sentiment analysis with a BoW with TF-IDF with *hyper parameter tuning*.

- `05_sa_word-embeddings-50d.ipynb`: training and testing a neural network for sentiment analysis with average embeddings built from 50 dimension word embedding pre-trained model.

- `06_sa_word-embeddings_200d.ipynb`: training and testing a neural network for sentiment analysis with average embeddings built from 200 dimension word embedding pre-trained model.

- `07_sa_word-embeddings.ipynb`: training and testing a neural network for sentiment analysis with average embeddings built from 2- dimension word embedding pre-trained model.


