from __future__ import print_function

import argparse

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import tensorflow as tf


from model import Model
from utils import *
from tensorboard_evaluation import Evaluation

def read_data(datasets_dir="./data", frac = 0.25):
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


    print("y.shape:", y.shape)

    j = 0


    for i in range(30000):
        if all(y[i] == [0., 0. , 0.]):
            if i > 17500:
                continue
            X[j] = X[i]
            y[j] = [0., 1., 0.]
            if i % 4 == 0 :
                y[j] = [0., 0., 0.2]    
            j += 1
        else:
            X[j] = X[i]
            y[j] = y[i]
            j += 1

    X = np.append(X, X[:5000,:,:], axis = 0)
    y = np.append(y, y[:5000,:], axis = 0)

    print("y.shape:", y.shape)

    

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]


    ##for i in range(10000):
    ##    print("\n\nIn Read y_train[",i,"]:", y_train[i])
    ##    i += 103

        
    return X_train, y_train, X_valid, y_valid


def count_output_data_hot_instances(y, i = ''):

    count_straight = 0
    count_left = 0
    count_right = 0
    count_acc = 0
    count_break = 0

    for i in range(y.shape[0]):
        if (y_train[i] == [1., 0., 0., 0.]):
            count_straight += 1
        if (y_train[i] == [0., 1., 0., 0.]):
            count_left += 1
        if (y_train[i] == [0., 0., 1., 0.]):
            count_right += 1
        if (y_train[i] == [0., 0., 0., 1.]):
            count_acc += 1
        if (y_train[i] == [0., 0., 0., 0.2]):
            count_break += 1

        print("\nStraight:",count_straight, "\nLeft:",count_left, "\nRight:",count_right,  "\nAccelerate:",count_acc, "\nBreak:",count_break)

            



def reshaped_history(x, history_length):

    print("Shape of x", x.shape)
    reshaped = np.empty((x.shape[0] - history_length + 1, x.shape[1], x.shape[2], history_length))
    print("Shape of Reshaped", reshaped.shape)
    #print("x:",x)

    for index in range(x.shape[0] - history_length):
        reshaped[index, :, :, :] = np.transpose(x[index: index + history_length, :, :, 0], (1, 2, 0))

    return reshaped

def reshaped_history_y(y, history_length):

    print("Shape of y", y.shape)
    reshaped = np.empty((y.shape[0] - history_length + 1, y.shape[1], history_length))
    print("Shape of Reshaped", reshaped.shape)
    #print("y:",y)

    for index in range(y.shape[0] - history_length):
        reshaped[index, :, :] = np.transpose(y[index: index + history_length,  0], ( 0))

    return reshaped


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

    print("\n\n\n X_train[10]", X_train[10])
    print("\n\n\n X_valid[10]", X_valid[10])
    
    X_valid = X_valid.reshape(X_valid.shape[0], 96, 96)
    
    X_train = X_train.reshape(X_train.shape[0], 96, 96)

    print("\n\n\n X_train[10]", X_train[10])
    print("\n\n\n X_valid[10]", X_valid[10])
    
    #y_train = y_train.astype('int32')

    y_train_id = np.zeros(y_train.shape[0], dtype = int)
    y_valid_id = np.zeros(y_valid.shape[0], dtype = int)


    for i in range(X_train.shape[0]):
        y_train_id[i] = action_to_id(y_train[i])

    for i in range(X_valid.shape[0]):
        y_valid_id[i] = action_to_id(y_valid[i])


    for i in range(10000):
        print("\n\ny_train[",i,"]:", y_train[i])
        print("\n\ny_train_id[",i,"]:", y_train_id[i])
        i += 103
    y_train = one_hot(y_train_id)
    y_valid = one_hot(y_valid_id)
    
    print('... done loading data')

    if history_length > 1 :
        X_train = reshaped_history(X_train, history_length)
        
        
        X_valid = reshaped_history(X_valid, history_length)
        y_train = reshaped_history_y(y_train, history_length)
        #print("Shape of y_train", y_train.shape)
        #y_train[history_length:] = y_train[history_length - 1:]
        y_valid = reshaped_history_y(y_valid, history_length)
        #y_valid[history_length:] = y_valid[history_length - 1:]

    '''SA'''
    
    return X_train, y_train, X_valid, y_valid



def train_model(X_train, y_train, X_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    tensorboard_eval = Evaluation(tensorboard_dir)

    # TODO: specify your neural network in model.py 
    agent = Model(batch_size = batch_size, lr = lr, history_length = history_length)
    print("... model created")

    train_accuracy = 0
    valid_accuracy = 0
    
    with agent.sess:
        for epoch in range(n_minibatches):
            _loss = 0
            for i in range(X_train.shape[0] // batch_size):
                first_index = np.random.randint(0, X_train.shape[0] - batch_size - history_length - 1)
                last_index = first_index + batch_size

                X_train_mini = X_train[first_index : last_index, :, :]
                y_train_mini = y_train[first_index : last_index, :]

                opt, l = agent.sess.run([agent.optimizer, agent.loss], feed_dict = {agent.x_image: X_train_mini, agent.y_: y_train_mini})
                _loss += l/(X_train.shape[0] //batch_size)

            train_accuracy = agent.accuracy.eval(feed_dict = {agent.x_image: X_train_mini, agent.y_: y_train_mini})

            print("\n\nPREDICTION:", agent.predict.eval(feed_dict={agent.x_image: X_train_mini}))
            valid_accuracy = 0

            for i in range(X_valid.shape[0] // batch_size):
                first_index = np.random.randint(0, X_train.shape[0] - batch_size - history_length - 1)
                last_index = first_index + batch_size

                X_valid_mini = X_valid[first_index : last_index, :, :]
                y_valid_mini = y_valid[first_index : last_index, :]

                ac = agent.accuracy.eval(feed_dict={agent.x_image: X_valid_mini, agent.y_: y_valid_mini})
                #print("ac:", ac)                    
                valid_accuracy += ac /(X_valid.shape[0] //batch_size)
                

            eval_dict = {"train":train_accuracy, "valid":valid_accuracy, "loss":_loss}
            tensorboard_eval.write_episode_data(epoch, eval_dict)

            print("Epoch:",epoch+1, "Train accuracy:", train_accuracy, "validation accuracy:", valid_accuracy, "loss:", _loss) 

        save_path = os.path.join(model_dir, "agent.ckpt")
        agent.save(save_path)

    tensorboard_eval.close_session()
    print("Model saved in file: %s" % model_dir)

    
   

    '''SA'''


if __name__ == "__main__":

    cmdline_parser = argparse.ArgumentParser('exercise3_R_NR')


    cmdline_parser.add_argument(
        '-l', '--history_length', default=1,
        help='History Length', type=int)
    args, unknowns = cmdline_parser.parse_known_args()

    history_length = args.history_length


    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length = history_length)
    #count_output_data_hot_instances(y_train)
    #count_output_data_hot_instances(y_valid)

    print("... preprocessing done")

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, n_minibatches=100, batch_size=64, lr=0.01)
 
