import tensorflow as tf
import numpy as np

class Model:
    
    def __init__(self, history_length, lr = 0.001, batch_size = 64):
        
        # TODO: Define network
        # ...
        '''SA'''
        #num_filters = 15
        #filter_size = 5
        with tf.name_scope("inputs"):
            self.x_image = tf.placeholder(tf.float32, shape=[None,96,96, history_length], name = "x_image")
            reshape_x = tf.reshape(self.x_image, shape = [-1, 96, 96, history_length], name = "reshape_x")
            self.y_ = tf.placeholder(tf.float32, shape=[None, 5], name = "y_")
        #y_conv = tf.placeholder(tf.float32, shape=[None, 10], name= 'y_conv')

        # first layers
        self.W_conv1 = tf.get_variable("W_conv1", [8, 8, history_length, batch_size], initializer=tf.contrib.layers.xavier_initializer())
        conv1 = tf.nn.conv2d(reshape_x, self.W_conv1, strides=[1, 2, 2, 1], padding='VALID')
        conv1_a = tf.nn.relu(conv1)
        pool1 = self.max_pool_2x2(conv1_a)

        # second layer
        self.W_conv2 = tf.get_variable("W_conv2", [4, 4, batch_size, batch_size], initializer=tf.contrib.layers.xavier_initializer())
        conv2 = tf.nn.conv2d(pool1, self.W_conv2, strides=[1, 2, 2, 1], padding='VALID')
        conv2_a = tf.nn.sigmoid(conv2)
        pool2 = self.max_pool_2x2(conv2_a)

        # third layer
        self.W_conv3 = tf.get_variable("W_conv3", [3, 3, batch_size, batch_size], initializer=tf.contrib.layers.xavier_initializer())
        conv3 = tf.nn.conv2d(pool2, self.W_conv3, strides=[1, 2, 2, 1], padding='VALID')
        conv3_a = tf.nn.relu(conv3)
        pool3 = self.max_pool_2x2(conv3_a)

        
        '''
        # fourth layer
        self.W_conv4 = tf.get_variable("W_conv4", [2, 2, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
        conv4 = tf.nn.conv2d(pool3, self.W_conv4, strides=[1, 2, 2, 1], padding='VALID')
        conv4_a = tf.nn.sigmoid(conv4)
        pool4 = self.max_pool_2x2(conv4_a)

        # fifth layer
        self.W_conv5 = tf.get_variable("W_conv5", [4, 4, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
        conv5 = tf.nn.conv2d(pool4, self.W_conv5, strides=[1, 2, 2, 1], padding='VALID')
        conv5_a = tf.nn.relu(conv5)
        pool5 = self.max_pool_2x2(conv5_a)
        '''
        

        flatten = tf.reshape(pool3, shape = [-1, 1*8*8])

        fc1 = tf.contrib.layers.fully_connected(flatten, 1024, activation_fn=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 512, activation = tf.nn.sigmoid, name = "fc2")
        fc3 = tf.contrib.layers.fully_connected(fc2, 256, activation_fn=tf.nn.sigmoid)

        '''

        a_lstm = tf.nn.rnn_cell.LSTMCell(num_units=256)
        a_lstm = tf.nn.rnn_cell.DropoutWrapper(a_lstm, output_keep_prob=1.0)
        a_lstm = tf.nn.rnn_cell.MultiRNNCell(cells=[a_lstm])

        a_init_state = a_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
        lstm_in = tf.expand_dims(fc3, axis=1)

        a_outputs, a_final_state = tf.nn.dynamic_rnn(cell=a_lstm, inputs=lstm_in, initial_state=a_init_state)

        a_cell_out = tf.reshape(a_outputs, [-1, 256], name='flatten_lstm_outputs')
        '''
        

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(fc3, 5, name = "output")
            y_pred = tf.nn.softmax(self.logits, name = "y_pred")
            self.predict = tf.argmax(self.logits, 1)
            #print("\n\nPREDICTION:", self.predict)

            

        # TODO: Loss and optimizer
        with tf.name_scope("train"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optimizer = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)
            
            
        with tf.name_scope("eval"):
            self.prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y_, 1))
            #prediction = tf.argmax(self.logits, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.prediction, tf.float32))

            

            
        # TODO: Start tensorflow session
        with tf.name_scope("init_and_save"):
            init = tf.global_variables_initializer()
        
        self.sess = tf.Session();
        self.sess.run(init)
        self.saver = tf.train.Saver()

        



        '''
        #batch_size = tf.shape(self.x_image)[0]

##        # sixth layer
##        self.W_conv6 = tf.get_variable("W_conv6", [4, 4, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
##        conv6 = tf.nn.conv2d(conv5_a, self.W_conv6, strides=[1, 2, 2, 1], padding='VALID')
##        conv6_a = tf.nn.sigmoid(conv6)

        


        flatten = tf.contrib.layers.flatten(conv5_a)
        # first dense layer + relu + dropout
        fc1 = tf.contrib.layers.fully_connected(flatten, 400, activation_fn=tf.nn.relu)
        fc1_drop = tf.nn.dropout(fc1, 1.0)
        # second dense layer + sigmoid + dropout:
        fc2 = tf.contrib.layers.fully_connected(fc1_drop, 400, activation_fn=tf.nn.sigmoid)
        fc2_drop = tf.nn.dropout(fc2, 1.0)
        # third dense layer + relu
        fc3 = tf.contrib.layers.fully_connected(fc2_drop, 100, activation_fn=tf.nn.relu)

        # LSTM layer
        a_lstm = tf.nn.rnn_cell.LSTMCell(num_units=256)
        a_lstm = tf.nn.rnn_cell.DropoutWrapper(a_lstm, output_keep_prob=1.0)
        a_lstm = tf.nn.rnn_cell.MultiRNNCell(cells=[a_lstm])

        a_init_state = a_lstm.zero_state(batch_size=batch_size, dtype=tf.float64)
        lstm_in = tf.expand_dims(fc3, axis=1)

        a_outputs, a_final_state = tf.nn.dynamic_rnn(cell=a_lstm, inputs=lstm_in, initial_state=a_init_state)
        if history_length > 1:
            a_outputs = self.reshaped_history(a_outputs, history_length)
        a_cell_out = tf.reshape(a_outputs, [-1, 256], name='flatten_lstm_outputs')


        # TODO: Loss and optimizer
        # ...       

        self.output = tf.contrib.layers.fully_connected(a_cell_out, 4, activation_fn=None)
        #self.output = self.reshaped_history_y(self.output, history_length)
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=self.output, labels= self.y_))

        
        #cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
        
        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.y_ , logits = self.cost)
        self.cross_entropy = tf.reduce_mean(self.cross_entropy)*100
        self.train_step = tf.train.GradientDescentOptimizer(lr).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        

        
        self.prediction = tf.argmax(self.y_conv,1, name='prediction')
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='accuracy')
        

        self.optimizer = tf.train.GradientDescentOptimizer(lr).minimize(self.cost)

        self.sess = tf.Session()
        #self.sess = tf.InteractiveSession()
        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()
        
        '''
    
    

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


    def reshaped_history(self, x, history_length):
        print("Shape of x", x.shape)
        reshaped = np.empty((x.shape[0] - history_length + 1, x.shape[1], x.shape[2], history_length))
        print("Shape of Reshaped", reshaped.shape)
        #print("x:",x)

        for index in range(x.shape[0] - history_length):
            reshaped[index, :, :, :] = np.transpose(x[index: index + history_length, :, :, 0], (1, 2, 0))

        return reshaped

        
