#!/usr/bin/env python

# The code is used to predict the stock prices using the EMD2FNN presented 
# in the following paper.
# -- Feng Zhou, Haomin Zhou, Zhihua Yang, Lihua Yang, EMD2FNN: A strategy 
# -- combining empirical mode decomposition and factorization machine based
# -- neural network for stock market trend prediction. Expert Systems with 
# -- Applications, 2019, 115:136-151.

# Coder: Feng Zhou (fengzhou@gdufe.edu.cn)
# Wrote in Nov. 2017
# Revised in June 2019


import math
import numpy as np


# Helper function to evaluate the total loss on the dataset
# -- model: the learnt FNN model
# -- X: features of the samples
# -- Y: labels of the samples
def calculate_loss(model, X, Y):
    
    num_examples = X.shape[0]
    inter_weight, W1, b1, inter_weight2, W2, b2, W3, b3= model['inter_weight'], \
        model['w1'], model['b1'], model['inter_weight2'], model['w2'], model['b2'], \
        model['w3'], model['b3']
    linear_num = W1.shape[1]
    
    z1 = np.zeros((X.shape[0], inter_weight.shape[1] + linear_num))
    z1[:, :linear_num] = X.dot(W1) + b1
    z1[:, linear_num : ] = 0.5 * (np.square(X.dot(inter_weight)) \
        - np.square(X).dot(np.square(inter_weight))) 
    a1 = np.tanh(z1)  
    
    z2 = np.zeros((X.shape[0], inter_weight2.shape[1] + linear_num))
    z2[:, :linear_num] = a1.dot(W2) + b2
    z2[:, linear_num : ] = 0.5 * (np.square(a1.dot(inter_weight2)) - \
        np.square(a1).dot(np.square(inter_weight2))) 
    a2 = np.tanh(z2) 
    
    z3 = a2.dot(W3) + b3
    probs = z3 
    
    # the loss layer
    data_loss = np.sum(np.power(Y - probs, 2))
    
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + \
        np.sum(np.square(W3)) + reg_ratio * (np.sum(np.square(inter_weight)) + \
        reg_ratio2 * np.sum(np.square(inter_weight2))))
    
    return data_loss / num_examples


# Helper function to predict an output
# -- model: the learnt FNN model
# -- x_sample: features of the samples
def predict(model, x_sample):
    
    inter_weight, W1, b1, inter_weight2, W2, b2, W3, b3= model['inter_weight'], \
        model['w1'], model['b1'], model['inter_weight2'], model['w2'], model['b2'], \
        model['w3'], model['b3']
    linear_num = W1.shape[1]
    
    z1 = np.zeros((x_sample.shape[0], inter_weight.shape[1] + linear_num))
    z1[:, :linear_num] = x_sample.dot(W1) + b1
    z1[:, linear_num : ] = 0.5 * (np.square(x_sample.dot(inter_weight)) - \
        np.square(x_sample).dot(np.square(inter_weight))) 
    a1 = np.tanh(z1) 
    
    z2 = np.zeros((x_sample.shape[0], inter_weight2.shape[1] + linear_num))
    z2[:, : linear_num] = a1.dot(W2) + b2
    z2[:, linear_num : ] =  0.5 * (np.square(a1.dot(inter_weight2)) - \
        np.square(a1).dot(np.square(inter_weight2)))
    a2 = np.tanh(z2)
    
    z3 = a2.dot(W3) + b3
    probs = z3
    
    return probs


# This function learns parameters for the neural network and returns the model.
# -- X: Features of the samples
# -- Y: Labels of the samples
# -- nn_hdim: The number of the neurons of the 1st hidden layer
# -- num_passes: Number of passes through the training data for gradient descent
# -- epsilon: the initial learning rate in SGD
# -- print_loss: If True, print the loss every 2000 iterations
def build_model(X, Y, nn_hdim, num_passes=2000, epsilon=1e-4, print_loss=False):
    
    np.random.seed(0)
    linear_num = 1
    hidden_2num = int(nn_hdim / 2) #set the number of the neurons of the 2nd hidden layer
    nn_input_dim = X.shape[1]
    num_examples = X.shape[0]
    nn_output_dim = Y.shape[1]
    
    # Initialize the parameters to random values
    w1 = np.random.randn(nn_input_dim, linear_num) / np.sqrt(nn_input_dim)
    inter_weight = np.random.randn(nn_input_dim, nn_hdim - linear_num) / \
        np.sqrt(nn_input_dim)
    b1 = np.zeros((1, linear_num))
    w2 = np.random.randn(nn_hdim, linear_num) / np.sqrt(nn_hdim)
    inter_weight2 = np.random.randn(nn_hdim, hidden_2num - linear_num) / \
        np.sqrt(nn_hdim)
    b2 = np.zeros((1, linear_num))
    w3 = np.random.randn(hidden_2num, nn_output_dim) / np.sqrt(hidden_2num)
    b3 = np.zeros((1, nn_output_dim))
      
    # Initialize the outputs in the 1st and 2nd hidden layers
    z1 = np.zeros((num_examples, nn_hdim))
    z2 = np.zeros((num_examples, hidden_2num))
    
    #This is used to save the learnt model
    model = {}

    # gradient descent. for each batch...
    old_loss = 1e6
    for iteration in xrange(1, num_passes+1):
        
        # update the learning rate in SGD (optional) 
        #if iteration > 1e4 and iteration % 2000 == 0:
        #    epsilon /= 2
        #else:
        #    epsilon = epsilon

        # forward propagation
        # -- the 1st hidden layer
        z1[:, : linear_num] = X.dot(w1) + b1
        z1[:, linear_num :] = 0.5 * (np.square(X.dot(inter_weight)) - \
            np.square(X).dot(np.square(inter_weight))) 
        a1 = np.tanh(z1) # nonlinear activation function(also can be changed by the others) 
        
        # -- the 2nd hidden layer
        z2[:, 0:linear_num] = a1.dot(w2) + b2
        z2[:, linear_num : hidden_2num] = 0.5 * (np.square(a1.dot(inter_weight2)) - \
            np.square(a1).dot(np.square(inter_weight2)))
        a2 = np.tanh(z2) # nonlinear activation function(also can be changed by the others)
        
        # -- the output layer
        z3 = a2.dot(w3) + b3
        probs = z3 #it can be changed by sigmoid(z3) when we predict the rise or fall of stock price
        
        # backward propagation
        delta4 = (probs - Y)
        dw3 = np.dot(a2.T, delta4)
        db3 = np.sum(delta4, axis=0)
        
        delta3 = delta4.dot(w3.T) * (1 - np.power(a2, 2))
        dw2 = (a1.T).dot(delta3[:, 0:linear_num])
        db2 = np.sum(delta3[:, 0:linear_num], axis=0)
        dinter_weight2 = np.dot(a1.T, (a1.dot(inter_weight2)) * delta3[:, linear_num:]) - \
            np.dot(np.square(a1).T, delta3[:, linear_num:]) * inter_weight2
        
        delta2 = (np.dot((a1.dot(inter_weight2)) * delta3[:, linear_num:], inter_weight2.T) \
            - delta3[:, linear_num:].dot(np.square(inter_weight2).T) * a1 + np.dot(delta3[:, \
            0:linear_num], w2.T)) * (1 - np.power(a1, 2)) 
        dw1 = np.dot(X.T, delta2[:, 0:linear_num])
        db1 = np.sum(delta2[:, 0:linear_num], axis=0)
        dinter_weight = np.dot(X.T, (X.dot(inter_weight)) * delta2[:, linear_num:nn_hdim]) - \
            np.dot(np.square(X).T, delta2[:, linear_num:nn_hdim]) * inter_weight

        # add regularization terms (b1, b2 and b3 don't have regularization terms)
        dinter_weight +=  reg_lambda * reg_ratio * inter_weight
        dinter_weight2 +=  reg_lambda * reg_ratio * reg_ratio2 * inter_weight2
        dw3 += reg_lambda * w3
        dw2 += reg_lambda * w2
        dw1 += reg_lambda * w1
         
        # gradient descent parameter update
        w1 += -epsilon * dw1
        inter_weight += -epsilon * dinter_weight
        b1 += -epsilon * db1
        w2 += -epsilon * dw2
        inter_weight2 += -epsilon * dinter_weight2
        b2 += -epsilon * db2
        w3 += -epsilon * dw3
        b3 += -epsilon * db3
                 
        # assign new parameters to the model
        model = {'inter_weight': inter_weight, 'w1': w1,'b1': b1, 'inter_weight2': \
            inter_weight2, 'w2': w2, 'b2': b2, 'w3': w3, 'b3': b3}
        
        # optionally print the loss.
        # this is expensive because it uses the whole dataset, so we don't want to do it too often.
        loss = calculate_loss(model, X, Y)
        if print_loss and iteration % 2000 == 0:
                print "Loss after iteration: %i, loss: %.5f" %(iteration, loss)
        
    return model


# read the data and construct the training, validation and test datasets
def construct_datasets(file_name, output_len, wnd_size):
    
    data = np.loadtxt(open(file_name, 'rb'), delimiter=',', skiprows=0)
    
    min_data = np.min(data)
    max_data = np.max(data)
    norm_data = (data - min_data) / (max_data - min_data)
    sum_data = np.sum(norm_data, axis = 1)
    
    data_len = sum_data.shape[0]
    sample_num = data_len -  wnd_size - output_len + 1
    
    # construct the samples from the time series's IMFs
    feats_num = wnd_size * data.shape[1]
    X = np.zeros((sample_num, feats_num + output_len))
    for i in range(sample_num):
        X[i, : -output_len] = norm_data[i : i + wnd_size, :].reshape(feats_num)
        X[i, - output_len :] = sum_data[i + wnd_size : i + wnd_size + output_len]
    
    #np.random.shuffle(X) # shuffle the samples (optional)
    
    # divide the samples into training, validation and test datases as the ratio 7:2:1
    train_num = int(math.floor(0.7 * sample_num))
    validate_num = int(math.floor(0.2 * sample_num))
    test_num = int(math.floor(0.1 * sample_num))
    train_X = X[: train_num, : - output_len]
    validate_X = X[train_num : train_num + validate_num, : - output_len]
    test_X = X[train_num + validate_num :, : - output_len]
    train_Y = X[: train_num, - output_len :]
    validate_Y = X[train_num : train_num + validate_num, - output_len :]
    test_Y = X[train_num + validate_num :, - output_len :]
    
    return norm_data, max_data, min_data, train_X, train_Y, validate_X, \
        validate_Y, test_X, test_Y

# read the samples obtained from the scrolling decomposition and 
# divide them into training, validation and test datasets
def construct_datasets_scrolling(file_Feats, file_Labels):
    
    X = np.loadtxt(open(file_Feats, 'rb'), delimiter=',', skiprows=0)
    Y = np.loadtxt(open(file_Labels, 'rb'), delimiter=',', skiprows=0)
    
    
    sample_num = X.shape[0]
    output_len = Y.shape[1]

    min_data = np.min(X)
    max_data = np.max(X)
    norm_data = (X - min_data) / (max_data - min_data)
    Y = (Y - min_data) / (max_data - min_data)

    
    # divide the samples into training, validation and test datases as the ratio 7:2:1
    train_num = int(math.floor(0.7 * sample_num))
    validate_num = int(math.floor(0.2 * sample_num))
    test_num = int(math.floor(0.1 * sample_num))
    
    train_X = X[: train_num, :]
    validate_X = X[train_num : train_num + validate_num, :]
    test_X = X[train_num + validate_num :, :]
    train_Y = Y[: train_num, :]
    validate_Y = Y[train_num : train_num + validate_num, :]
    test_Y = Y[train_num + validate_num :, :]
    
    return norm_data, max_data, min_data, train_X, train_Y, validate_X, \
        validate_Y, test_X, test_Y
    

if __name__ == "__main__":
    
    np.random.seed(0)
    
    # set the file's directory and its name
    file_name = './data/lod_imf.csv'
    
    #file_Feats = './data/lod_Feats.csv'
    #file_Labels = './data/lod_Feats.csv'
    
    # set the predicion length
    output_len = 1

    # set the window lenght
    wnd_size = 3 
    
    # read the data and construct the training, validation and test datasets
    norm_data, max_data, min_data, train_X, train_Y, validate_X, validate_Y, test_X, test_Y = \
        construct_datasets(file_name, output_len, wnd_size)
    
    #norm_data, max_data, min_data, train_X, train_Y, validate_X, validate_Y, test_X, test_Y = \
    #   construct_datasets_scrolling(file_Feats, file_Labels)
    
    
    old_loss =1e6
    # selecting the optimal hyper-parameters by grid search (I picked these by hand.) 
    # Notice: this process is time consuming as the ranges of hyper-parameters increase.
    for epsilon in range(1, 100, 5): # initial learning rate for gradient descent
        epsilon *= 0.00005
        for reg_lambda in [0.05, 0.1, 0.2]: #regularization strength
            reg_lambda *= 0.001
            for nnhid_num in range(10, 200, 10): # the neurons size of the 1st hidden layer
                for reg_ratio in [0.001, 0.01, 0.1, 1]: # the additional regularization strength on the 1st FM weights
                    for reg_ratio2 in [1]: # the additional regularization strength on the 2nd FM weights
                            
                            # train
                            model = build_model(train_X, train_Y, nnhid_num, 20000, epsilon, print_loss=True)
                            
                            # search the optimal FNN model on validate dataset
                            Pred = predict(model, validate_X)
                            Pred = Pred * (max_data - min_data) + min_data * norm_data.shape[1]
                            loss = np.mean(np.abs(validate_Y - Pred) / np.abs(validate_Y)) # MAPE index
                            if loss < old_loss:
                                final_model = model
                                old_loss = loss
                                print "epsilon: %.8f, reg_lambda: %.8f, nnhid_num: %d, reg_ratio: %.5f, loss: %.8f" \
                                    %(epsilon, reg_lambda, nnhid_num, reg_ratio, loss)

    print "Traning FNN is done!"
    
    # predict on test dataset
    Pred = predict(final_model, test_X)
    Pred = Pred * (max_data - min_data) + min_data * norm_data.shape[1]
    print "FNN predictions:", Pred
