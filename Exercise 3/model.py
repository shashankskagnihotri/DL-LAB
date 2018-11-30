import tensorflow as tf

class Model:
    
    def __init__(self):
        
        # TODO: Define network
        # ...
        '''SA'''
        num_filters = 15
        filter_size = 5

        x_image = tf.placeholder(tf.float32, shape=[None,96,96,1], name = "x_image")
        y_ = tf.placeholder(tf.float32, shape=[None, 4], name = "y_")
        #y_conv = tf.placeholder(tf.float32, shape=[None, 10], name= 'y_conv')

        print("x_image:", x_image.shape)
        print("y_:", y_.shape)

        W_conv1 = self.weight_variable([filter_size, filter_size, 1, num_filters])
        print("W_conv1:", W_conv1.shape)
        b_conv1 = bias_variable([num_filters])
        print("b_conv1:", b_conv1.shape)

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1 , name = "h_conv1")
        h_pool1 = self.max_pool_2x2(h_conv1)
        W_conv2 = selfweight_variable([filter_size, filter_size, num_filters, num_filters])
        b_conv2 = self.bias_variable([num_filters])

        print("h_conv1:", h_conv1.shape)
        print("h_pool1:", h_pool1.shape)
        print("W_conv2:", W_conv2.shape)
        print("b_conv2:", b_conv2.shape)


        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2 , name = "h_conv2")
        h_pool2 = self.max_pool_2x2(h_conv2)
        W_fc1 = self.weight_variable([96 * 96 * num_filters, 4])
        b_fc1 = self.bias_variable([4])

        print("h_conv2:", h_conv2.shape)
        print("h_pool2:", h_pool2.shape)
        print("W_fc1:", W_fc1.shape)
        print("b_fc1:", b_fc1.shape)


        h_pool2_flat = tf.reshape(h_pool2, [-1, 96*96*num_filters] , name = "h_pool2")

        dense = tf.layers.dense(inputs = h_pool2_flat, units = 4, activation = tf.nn.relu , name = "dense")

        y_pred = tf.layers.dense(inputs = dense, units = 4, name = "y_pred")

        print("h_pool2flat:", h_pool2_flat.shape)
        print("y_pred:", y_pred.shape)

        
        
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1 , name = "h_fc1")
        keep_prob = tf.placeholder(tf.float32, name = "keep")
        #keep_prob = 0.5
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob , name = "h_f1_drop")

        W_fc2 = self.weight_variable([4, 4])
        b_fc2 = self.bias_variable([4])

        print("h_fc1:", h_fc1.shape)
        print("h_fc1_drop:", h_fc1_drop.shape)

        print("W_fc2:", W_fc2.shape)
        print("b_fc2:", b_fc2.shape)

        #y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2 , name = "y_conv")

        
        
        y_conv=tf.nn.softmax(W_fc2 + b_fc2 , name = "y_conv")
        print("y_conv:", y_conv.shape)

        n_samples = x_train.shape[0]
        n_batches = n_samples // batch_size

        # TODO: Loss and optimizer
        # ...       

        #cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = y_ , logits = y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)*100
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        saver = tf.train.Saver()
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

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)


    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        
