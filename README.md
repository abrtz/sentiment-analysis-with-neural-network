# sentiment-analysis-with-neural-network
Sentiment Analysis with a Neural Network

This repository contains the notebooks to build my own neural network to perform an NLP task. The task chosen is sentiment analysis.

The neural network was built following the steps described in the book Make Your Own Neural Network by Tariq Rashid (2016).

The complete dataset can be downloaded from: https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews?resource=download
The word2vec file can be downloaded from: https://www.kaggle.com/datasets/watts2/glove6b50dtxt
The .csv file in this repository contains the data after exploration, needed to run notebooks 05 onwards.

Notebooks 01 and 02 contain the code applied following the code proposed in the abovementioned book to create a NN for image recognition.

Notebooks 03 and 04 contain the code to explore the dataset to perform the sentiment analysis task.

Notebooks 05, 06 and 07 contain the code with the experiments carried out for the NLP task:
* 05: sentiment classification with a BoW vector counting bigrams with a minimum frequency of 300.
* 06: sentiment classification with a padded vector created from an n-gram BoW mapping the indices of words appearing in the vector to those appearing in each text.
* 07: sentiment classification with a word embedding padded vector.
* 08: hyperparameter tuning on neural network model with word embedding.
