from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import color from skimage

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

    X_train = utils.rgb2gray(X_train)
    X_valid = utils.rgb2gray(X_valid)

    
    X_valid = X_valid.astype('float32')
    X_valid = X_valid.astype('float32').reshape(X_valid.shape[0], 96, 96, 1)
    y_valid = y_valid.astype('int32')
    
    X_train = x_train.astype('float32').reshape(X_train.shape[0], 96, 96, 1)
    y_train = y_train.astype('int32')
    print('... done loading data')

    

    '''SA'''
    
    return X_train, y_train, X_valid, y_valid

'''SA'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
strides=[1, 2, 2, 1], padding='VALID')

'''SA'''

def train_model(X_train, y_train, X_valid, n_minibatches, batch_size, num_filters, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    # TODO: specify your neural network in model.py 
    # agent = Model(...)
    
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
    # model_dir = agent.save(os.path.join(model_dir, "agent.ckpt"))
    # print("Model saved in file: %s" % model_dir)

    '''SA'''

    x_image = tf.placeholder(tf.float32, shape=[None,96,96,1], name = "x_image")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name = "y_")
    #y_conv = tf.placeholder(tf.float32, shape=[None, 10], name= 'y_conv')

    W_conv1 = weight_variable([filter_size, filter_size, 1, num_filters])
    b_conv1 = bias_variable([num_filters])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([filter_size, filter_size, num_filters, num_filters])
    b_conv2 = bias_variable([num_filters])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1 = weight_variable([7 * 7 * num_filters, 128])
    b_fc1 = bias_variable([128])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*num_filters])

    dense = tf.layers.dense(inputs = h_pool2_flat, units = 128, activation = tf.nn.relu)

    y_pred = tf.layers.dense(inputs = dense, units = 10)

    
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32, name = "keep")
    #keep_prob = 0.5
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([128, 10])
    b_fc2 = bias_variable([10])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    #y_conv=tf.nn.softmax(W_fc2 + b_fc2)

    n_samples = x_train.shape[0]
    n_batches = n_samples // batch_size

   

    #cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = y_ , logits = y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)*100
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    saver = tf.train.Saver()
    learning_curve = np.zeros(num_epochs)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

    with tf.Session() as sess:
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        
        #learning_curve = np.zeros(num_epochs)
        for i in range(num_epochs):
            for b in range(n_batches):
                x_batch = x_train[b*batch_size:(b+1)*batch_size]
                y_batch = y_train[b*batch_size:(b+1)*batch_size]
                train_step.run(feed_dict={x_image: x_batch, y_: y_batch, keep_prob: 1.0})
                #train_step.run(feed_dict={x_image: x_batch, y_: y_batch})
                    
                #print("step %d, training accuracy %g"%(i, train_accuracy))
            #train_step.run(feed_dict={x_image: x_batch, y_: y_batch, keep_prob: 0.2})
            #train_accuracy = 1 - accuracy.eval(feed_dict={x_image:x_train , y_: y_train})
                
            learning_curve[i] = 1 - accuracy.eval(feed_dict={x_image: x_valid, y_: y_valid, keep_prob: 1.0})
            #learning_curve[i] = 1 - accuracy.eval(feed_dict={x_image: x_valid, y_: y_valid})
            print("step %d, train error %g"%(i, learning_curve[i]))
"""print("test accuracy %g"%accuracy.eval(feed_dict={x_image: x_valid, y_: y_valid, keep_prob: 0.2}))  """
    # TODO: save your agent
    model_dir = saver.save(sess, os.path.join(model_dir, "agent.ckpt")
    #model_dir = agent.save(os.path.join(model_dir, "agent.ckpt"))
    print("Model saved in file: %s" % model_dir)

    '''SA'''


if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, n_minibatches=100000, batch_size=64, num_filter=5, lr=0.0001)
 
