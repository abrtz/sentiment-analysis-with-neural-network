# Sentiment Analysis with a Neural Network

This repository contains the notebooks to build a neural network for sentiment analysis.

The neural network was built following the steps described in the book Make Your Own Neural Network by Tariq Rashid (2016).

## Project Structure:

```
.
├── LICENSE
└── neural_network
    ├── 01_sa_bow_unigrams.ipynb
    ├── 02_sa_bow_bigrams.ipynb
    ├── 03_sa_bow_trigrams.ipynb
    ├── 04_sa_bow_tfidf.ipynb
    ├── 05_sa_word-embeddings-50d.ipynb
    ├── 06_sa_word-embeddings_200dim.ipynb
    ├── 07_sa_word-embeddings.ipynb
    ├── data-exploration.ipynb
    ├── data-preparation.ipynb
    ├── models
    │   ├── nn_bigram_bow_model.pkl
    │   ├── nn_bigram_bow_vectorizer.pkl
    │   ├── nn_mean_we_model_200d.pkl
    │   ├── nn_mean_we_model_300d.pkl
    │   ├── nn_mean_we_model_50d.pkl
    │   ├── nn_tfidf_bow_model.pkl
    │   ├── nn_tfidf_bow_vectorizer.pkl
    │   ├── nn_trigram_bow_model.pkl
    │   ├── nn_trigram_bow_vectorizer.pkl
    │   ├── nn_unigram_bow_model.pkl
    │   └── nn_unigram_bow_vectorizer.pkl
    ├── nn.py
    ├── sephora-data
    │   ├── sa-reviews_dev.csv
    │   ├── sa-reviews_test.csv
    │   └── sa-reviews_training.csv
    ├── train_test.py
    ├── utils_data_exploration.py
    ├── utils_preprocess_data.py
    ├── utils_word_embeddings.py
    ├── utils_word_embeddings_50d.py
    └── utils_word_embeddings_glove.py
```

## Requirements

To run these experiments, make sure libraries required are installed. They are listed in the `requirements.txt` file.
The experiments were run in Python 3.11.8.
They can be installed with the following command:

`pip install -r requirements.txt`


## README

### Data

The complete dataset can be downloaded from: `https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews?resource=download` \
To run the data exploration and preparation, the complete dataset has to be downloaded and placed under the `/sephora-data` directory \

To run the neural network experiments, the preprocessed files are provided under the `/neural_network/sephora-data` subdirectory.

### Models
The word2vec file can be downloaded from: https://www.kaggle.com/datasets/watts2/glove6b50dtxt \
The Google news model can be downloaded from: https://github.com/mmihaltz/word2vec-GoogleNews-vectors \
Place both models file under the `/models` directory.


To run these experiments, the trained models are provided under the `/neural_network/models` subdirectory together with the vectorizers applied.

### Data exploration and preprocessing 

The notebooks `data-exploration.ipynb` and `data-preparation.ipynb` explore the original data files and prepare the dataset for the neural network experiments to perform the sentiment analysis task. 

### Experiments

- `01_sa_bow_unigrams.ipynb`: training and testing a neural network for sentiment analysis with a unigram BoW

- `02_sa_bow_bigrams.ipynb`: training and testing a neural network for sentiment analysis with a bigram BoW

- `03_sa_bow_trigrams.ipynb`: training and testing a neural network for sentiment analysis with a trigram BoW

- `04_sa_bow_tfidf.ipynb`: training and testing a neural network for sentiment analysis with a BoW with TF-IDF with *hyper parameter tuning*.

- `05_sa_word-embeddings-50d.ipynb`: training and testing a neural network for sentiment analysis with average embeddings built from 50 dimension word embedding pre-trained model.

- `06_sa_word-embeddings_200d.ipynb`: training and testing a neural network for sentiment analysis with average embeddings built from 200 dimension word embedding pre-trained model.

- `07_sa_word-embeddings.ipynb`: training and testing a neural network for sentiment analysis with average embeddings built from 2- dimension word embedding pre-trained model.


