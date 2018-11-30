from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import tensorflow as tf


from model import Model
from utils import *
from tensorboard_evaluation import Evaluation

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot() 
    #    useful and you may want to return X_train_unhot ... as well.

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).

    '''SA'''

    X_train = rgb2gray(X_train)
    X_train = np.expand_dims(X_train, axis=3)
    X_valid = rgb2gray(X_valid)
    X_valid = np.expand_dims(X_valid, axis=3)

    
    #X_valid = X_valid.astype('float32').reshape(X_valid.shape[0], 96, 96, 1)
    
    #X_train = x_train.astype('float32').reshape(X_train.shape[0], 96, 96, 1)
    #y_train = y_train.astype('int32')

    t = X_train.shape[0]
    v = X_valid.shape[0]

    y_train_id = np.zeros(t, dtype = int)
    y_valid_id = np.zeros(v, dtype = int)

    for i in range(t):
        y_train_id[i] = action_to_id(y_train[i])

    for i in range(v):
        y_valid_id[i] = action_to_id(y_valid[i])

    y_train = one_hot(y_train_id)
    y_valid = one_hot(y_valid_id)
    print('... done loading data')

    

    '''SA'''
    
    return X_train, y_train, X_valid, y_valid



def train_model(x_train, y_train, x_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    # TODO: specify your neural network in model.py 
    agent = Model()
    
    tensorboard_eval = Evaluation(tensorboard_dir)

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training in your web browser
    # 
    # training loop
    # for i in range(n_minibatches):
    #     ...
    #     tensorboard_eval.write_episode_data(...)
      
    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "agent.ckpt"))
    print("Model saved in file: %s" % model_dir)

    

   

    '''SA'''


if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, n_minibatches=100000, batch_size=64, lr=0.0001)
 
