import numpy as np
import time
import datetime
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt
from itertools import product
from nn import neuralNetwork

def format_time(elapsed):
    ### code taken from https://www.kaggle.com/code/themeeemul/sephora-eda-and-sentiment-analysis-using-pytorch/notebook on 26th Jan 
    ### useful to print how long the epochs take
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def calculate_accuracy(nn, features_vectorized, gold_labels_train):
    """
    Calculate and return the accuracy of the neural network on the training dataset.
    
    Parameters:
    - nn: an instance of the neural network class
    - features_vectorized: the training dataset (features)
    - gold_labels_train: the training gold labels

    """

    # initializing the variables to track correct predictions
    correct = 0
    total = features_vectorized.shape[0]  # total number of instances in the training dataset

    # find the maximum count in the entire matrix
    max_count = features_vectorized.max()

    # iterating over each instance in the training dataset
    for array, true_label in zip(features_vectorized, gold_labels_train):
        # normalize the inputs
        inputs = (array / float(max_count) * 0.99) + 0.01

        # query the neural network to get the output
        predicted_label = nn.query(inputs)

        # Convert the predicted label to binary (0 or 1) based on a threshold (e.g., 0.5)
        if (predicted_label >= 0.5 and true_label == 1) or (predicted_label < 0.5 and true_label == 0): #check if prediction matches the true label
            correct += 1 #threshold is set to 0.5

    # calculate the accuracy as the ratio of correct predictions to the total number of instances
    accuracy = correct / total
    
    return accuracy


def train_nn(nn, features_vectorized, gold_labels_train, epochs):
    """
    Train a neural network with the provided training dataset and gold labels.
    Return the trained neural network.

    Parameters:
    - nn: an instance of the neural network class
    - features_vectorized: the training dataset (features)
    - gold_labels_train: the training gold labels
    - epochs: number of times the training data set is ran for training
    """

    # find the maximum count in the entire matrix for normalization, integer given by the max number of an item in the vocabulary array
    max_count = features_vectorized.max()

    # lists to store accuracy values and epoch numbers
    accuracies = []
    epoch_numbers = []
    
    for epoch in range(epochs):
        # measure how long the training epoch takes
        start_time = time.time()
        
        #iterating over each training instance
        for array, value in zip(features_vectorized, gold_labels_train):
            # normalizing the inputs
            inputs = (array / float(max_count) * 0.99) + 0.01
        
            # creating the target output value (0.01 for negative sentiment, 0.99 for positive sentiment)
            if value == 1.0:
                target = 0.99  # Positive sentiment
            else:
                target = 0.01  # Negative sentiment
            #training the neural network
            nn.train(inputs, target)
            
        # measure elapsed time for the epoch
        elapsed_time = time.time() - start_time

        # calculate and store the accuracy after each epoch 
        accuracy = calculate_accuracy(nn, features_vectorized, gold_labels_train)
        accuracies.append(accuracy)
        epoch_numbers.append(epoch + 1)

        # print the epoch training time
        print(f"Epoch {epoch + 1}/{epochs}, Elapsed Time: {format_time(elapsed_time)}, Accuracy: {accuracy}")

    # plot the training progress
    plt.plot(epoch_numbers, accuracies, marker='o')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

    # return the trained neural network for saving the model
    return nn



def evaluate_nn(nn, data_test, g_labels_test):
    """
    Evaluate a neural network with the test or development data for validation.
    Print classification report with error metrics: precision, recall, F1 and accuracy.
    Plot the confusion matrix.

    Parameters:
    - nn: an instance of the neural network class
    - data_test: the test or dev dataset (features)
    - g_labels_test: the test or dev labels
    """

    # initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []
    labels = ['0','1'] #name of the labels for the classification report

    # find the maximum count in the entire matrix for normalization, integer given by the max number of an item in the vector
    max_count = data_test.max()
    
    # query the neural network for predictions on the test/dev data set
    for array, true_label in zip(data_test, g_labels_test):
        # normalizing the inputs
        inputs = (array / float(max_count) * 0.99) + 0.01
        # query the network
        predicted_label = nn.query(inputs)
        
        # convert the predicted label to binary (0 or 1) based on a threshold (e.g., 0.5)
        if predicted_label >= 0.5:
            predicted_label_binary = 1
        else:
            predicted_label_binary = 0
        
        # Append true label and predicted label to the lists
        true_labels.append(true_label)
        predicted_labels.append(predicted_label_binary)

    # Calculate classification report
    report = classification_report(true_labels, predicted_labels, digits=2,target_names=labels)
    cf_matrix = confusion_matrix(true_labels, predicted_labels)
    display = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=labels)
    print("Classification Report:")
    print(report)
    print(cf_matrix)
    display.plot()


def hyperparameter_tuning(input_nodes, output_nodes, feature_vec_train, gold_labels_train, feature_vec_dev, gold_labels_dev, hyperparameter_grid):
    """Perform hyperparameter tuning for a neural network model of an object class.
    Create all combinations of hyperparameters and create a neural network with each of them.
    Train the neural network for the number of epochs specified and evaluate the performance of the model on the development dataset.
    Identify the combination of hyperparameters that produced the highest accuracy and prints it.
    Return the best combination of hyperparameters and the results dictionary.

    Parameters:
    -'input_nodes' (int): the fixed number of input nodes of the nn.
    -'output_nodes' (int): the fixed number of input nodes of the nn.
    -'feature_vec_train' (np array): the feature vector of the training dataset.
    -'gold_labels_train' (np array): the training dataset gold labels.
    -'feature_vec_dev' (np array): the feature vector of the validation dataset.
    -'gold_labels_dev' (np array): the validation dataset gold labels.
    -'hyperparameter_grid' (dict): a dictionary containing hyperparameter names as keys and lists of possible values as values.
    """

    ###the code for this function was partially built with ChatGPT: product module from itertools.
    #creating all possible combinations of hyperparameters with the 'product' module from itertools
    param_combinations = list(product(*hyperparameter_grid.values()))
    
    results = dict() #an emtpy list where the results will be stored

    # find the maximum count in the entire matrix for normalization, integer given by the max number of an item in the vector
    max_count = feature_vec_train.max()
    
    #iterating over each combination of parameters
    for params in param_combinations:
        #unpacking the resulting parameters
        hidden_nodes, learning_rate, num_epochs = params

        #creating a neural network with the current hyperparameters
        nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

        #training the neural network with the training data
        for epoch in range(num_epochs):
            #iterating over each training instance
            for array, value in zip(feature_vec_train, gold_labels_train):
                # normalizing the inputs
                inputs = (array / float(max_count) * 0.99) + 0.01
            
                # creating the target output value (0.01 for negative sentiment, 0.99 for positive sentiment)
                if value == 1.0:
                    target = 0.99  # Positive sentiment
                else:
                    target = 0.01  # Negative sentiment
                #training the neural network
                nn.train(inputs, target)

        #evaluating its performance on the validation data
        accuracy = calculate_accuracy(nn, feature_vec_dev, gold_labels_dev)

        #storing the results to a dictionary with the combos as keys and the performance as values
        results[params] = accuracy
        
        print("Performance:", accuracy)
        print("Hidden nodes:", hidden_nodes)
        print("Learning rate:", learning_rate)
        print("Number of epochs:", num_epochs)   
        print()
        
    #finding the best set of hyperparameters based on the performance on dev data
    best_params = max(results, key=results.get)
    print("Best hyperparameters:",best_params)
    print("Best performance:", results[best_params])
    return best_params
