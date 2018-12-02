import tensorflow as tf
import numpy as np

class Model:
    
    def __init__(self, history_length, lr = 0.0002, batch_size = 64):
        
        # TODO: Define network
        # ...
        '''SA'''
        #num_filters = 15
        #filter_size = 5

        self.x_image = tf.placeholder(tf.float32, shape=[None,96,96,history_length], name = "x_image")
        self.y_ = tf.placeholder(tf.float32, shape=[None, 4], name = "y_")
        #y_conv = tf.placeholder(tf.float32, shape=[None, 10], name= 'y_conv')

        '''

        print("x_image:", self.x_image.shape)
        print("y_:", self.y_.shape)

        self.W_conv1 = self.weight_variable([filter_size, filter_size, 1, num_filters])
        print("W_conv1:", self.W_conv1.shape)
        self.b_conv1 = self.bias_variable([num_filters])
        print("b_conv1:", self.b_conv1.shape)

        self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1 , name = "h_conv1")
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)

        self.W_conv2 = self.weight_variable([filter_size, filter_size, num_filters, num_filters])
        self.b_conv2 = self.bias_variable([num_filters])

        print("h_conv1:", self.h_conv1.shape)
        print("h_pool1:", self.h_pool1.shape)
        print("W_conv2:", self.W_conv2.shape)
        print("b_conv2:", self.b_conv2.shape)


        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2 , name = "h_conv2")
        self.h_pool2 = self.max_pool_2x2(self.h_conv2)
        self.W_fc1 = self.weight_variable([96 * 96 * num_filters, 4])
        self.b_fc1 = self.bias_variable([4])

        print("h_conv2:", self.h_conv2.shape)
        print("h_pool2:", self.h_pool2.shape)
        print("W_fc1:", self.W_fc1.shape)
        print("b_fc1:", self.b_fc1.shape)


        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 96*96*num_filters] , name = "h_pool2")

        self.dense = tf.layers.dense(inputs = self.h_pool2_flat, units = 4, activation = tf.nn.relu , name = "dense")

        self.y_pred = tf.layers.dense(inputs = self.dense, units = 4, name = "y_pred")

        print("h_pool2flat:", self.h_pool2_flat.shape)
        print("y_pred:", self.y_pred.shape)

        
        
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1 , name = "h_fc1")
        self.keep_prob = tf.placeholder(tf.float32, name = "keep")
        #keep_prob = 0.5
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob , name = "h_f1_drop")

        self.W_fc2 = self.weight_variable([4, 4])
        self.b_fc2 = self.bias_variable([4])

        print("h_fc1:", self.h_fc1.shape)
        print("h_fc1_drop:", self.h_fc1_drop.shape)

        print("W_fc2:", self.W_fc2.shape)
        print("b_fc2:", self.b_fc2.shape)

        self.y_conv=tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2 , name = "y_conv")

        
        
        #self.y_conv=tf.nn.softmax(self.W_fc2 + self.b_fc2 , name = "y_conv")
        print("y_conv:", self.y_conv.shape)

        
        exit()
        self.n_samples = self.x_train.shape[0]
        self.n_batches = self.n_samples // batch_size
        '''

        #batch_size = tf.shape(self.x_image)[0]
        # first layers
        self.W_conv1 = tf.get_variable("W_conv1", [8, 8, history_length, 64], initializer=tf.contrib.layers.xavier_initializer())
        conv1 = tf.nn.conv2d(self.x_image, self.W_conv1, strides=[1, 2, 2, 1], padding='VALID')
        conv1_a = tf.nn.relu(conv1)

        # second layer
        self.W_conv2 = tf.get_variable("W_conv2", [4, 4, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
        conv2 = tf.nn.conv2d(conv1_a, self.W_conv2, strides=[1, 2, 2, 1], padding='VALID')
        conv2_a = tf.nn.sigmoid(conv2)

        # third layer
        self.W_conv3 = tf.get_variable("W_conv3", [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
        conv3 = tf.nn.conv2d(conv2_a, self.W_conv3, strides=[1, 2, 2, 1], padding='VALID')
        conv3_a = tf.nn.relu(conv3)

        # fourth layer
        self.W_conv4 = tf.get_variable("W_conv4", [4, 4, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
        conv4 = tf.nn.conv2d(conv3_a, self.W_conv4, strides=[1, 2, 2, 1], padding='VALID')
        conv4_a = tf.nn.sigmoid(conv4)

        # fifth layer
        self.W_conv5 = tf.get_variable("W_conv5", [4, 4, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
        conv5 = tf.nn.conv2d(conv4_a, self.W_conv5, strides=[1, 2, 2, 1], padding='VALID')
        conv5_a = tf.nn.relu(conv5)

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

        a_init_state = a_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
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
        '''
        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.y_ , logits = self.cost)
        self.cross_entropy = tf.reduce_mean(self.cross_entropy)*100
        self.train_step = tf.train.GradientDescentOptimizer(lr).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        '''

        '''
        self.prediction = tf.argmax(self.y_conv,1, name='prediction')
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='accuracy')
        '''

        self.optimizer = tf.train.GradientDescentOptimizer(lr).minimize(self.cost)

        #self.sess = tf.Session()
        self.sess = tf.InteractiveSession()
        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()



        '''
        num_epochs = n_minibatches
        learning_curve = np.zeros(num_epochs)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")


        # TODO: Start tensorflow session
        # ...

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
            model_dir = saver.save(sess, os.path.join(model_dir, "agent.ckpt"))
        #print('Model saved in file:')
        print(model_dir)
        
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

        
