import tensorflow as tf
import numpy as np

# TODO: add your Convolutional Neural Network for the CarRacing environment.

class NeuralNetwork():
    """
    Neural Network class based on TensorFlow.
    """
    def __init__(self, state_dim, num_actions, hidden=20, lr=1e-3):
        self._build_model(state_dim, num_actions, hidden, lr)
        
    def _build_model(self, state_dim, num_actions, hidden, lr):
        """
        This method creates a neural network with two hidden fully connected layers and 20 neurons each. The output layer
        has #a neurons, where #a is the number of actions and has linear activation.
        Also creates its loss (mean squared loss) and its optimizer (e.g. Adam with a learning rate of 1e-4).
        """

        self.states_ = tf.placeholder(tf.float32, shape=[None, state_dim])
        self.actions_ = tf.placeholder(tf.int32, shape=[None])                  # Integer id of which action was selected
        self.targets_ = tf.placeholder(tf.float32,  shape=[None])               # The TD target value

        # network
        fc1 = tf.layers.dense(self.states_, hidden, tf.nn.relu)
        fc2 = tf.layers.dense(fc1, hidden, tf.nn.relu)
        self.predictions = tf.layers.dense(fc2, num_actions)

        # Get the predictions for the chosen actions only
        batch_size = tf.shape(self.states_)[0]
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.targets_, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, sess, states):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """
        prediction = sess.run(self.predictions, { self.states_: states })
        return prediction


    def update(self, sess, states, actions, targets):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.
        
        Args:
          sess: TensorFlow session.
          states: [current_state] or states of batch
          actions: [current_action] or actions of batch
          targets: [current_target] or targets of batch
        """
        feed_dict = { self.states_: states, self.targets_: targets, self.actions_: actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class TargetNetwork(NeuralNetwork):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """
    def __init__(self, state_dim, num_actions, hidden=20, lr=1e-4, tau=0.01):
        NeuralNetwork.__init__(self, state_dim, num_actions, hidden, lr)
        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):
        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        op_holder = []
        for idx,var in enumerate(tf_vars[0:total_vars//2]):
            op_holder.append(tf_vars[idx+total_vars//2].assign(
              (var.value()*self.tau) + ((1-self.tau)*tf_vars[idx+total_vars//2].value())))
        return op_holder
      
    def update(self, sess):
        for op in self._associate:
          sess.run(op)

class CNN():
    def __init__(self, state_dim, num_actions, history_length = 3, hidden = 256, lr = 1e-3):
        self._build_model(state_dim, num_actions, history_length, hidden, lr)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    def _build_model(self, state_dim, num_actions, history_length, hidden, lr):
        self.states_ = tf.placeholder(tf.float32, shape=[None, *state_dim, history_length+1])
        self.actions_ = tf.placeholder(tf.int32, shape=[None])
        self.targets_ = tf.placeholder(tf.float32, shape=[None])

        # first layers
        W_conv1 = tf.get_variable("W_conv1", [8, 8, history_length, 64], initializer=tf.contrib.layers.xavier_initializer())
        conv1 = tf.nn.conv2d(reshape_x, W_conv1, strides=[1, 2, 2, 1], padding='VALID')
        conv1_a = tf.nn.relu(conv1)
        pool1 = max_pool_2x2(conv1_a)

        # second layer
        W_conv2 = tf.get_variable("W_conv2", [4, 4, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
        conv2 = tf.nn.conv2d(pool1, W_conv2, strides=[1, 2, 2, 1], padding='VALID')
        conv2_a = tf.nn.sigmoid(conv2)
        pool2 = max_pool_2x2(conv2_a)

        # third layer
        W_conv3 = tf.get_variable("W_conv3", [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
        conv3 = tf.nn.conv2d(pool2, W_conv3, strides=[1, 2, 2, 1], padding='VALID')
        conv3_a = tf.nn.relu(conv3)
        pool3 = max_pool_2x2(conv3_a)

        flatten = tf.reshape(pool3, shape = [-1, 1*8*8])

        fc1 = tf.contrib.layers.fully_connected(flatten, 1024, activation_fn=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 512, activation = tf.nn.sigmoid, name = "fc2")
        fc3 = tf.contrib.layers.fully_connected(fc2, 256, activation_fn=tf.nn.sigmoid)

        
        self.predictions = tf.layers.dense(fc3, num_actions)

         # Get the predictions for the chosen actions only
        batch_size = tf.shape(self.states_)[0]
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.targets_, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.GradientDescentOptimizer(lr)
        self.train_op = self.optimizer.minimize(self.loss)

        def predict(self, sess, states):
            """
            Args:
              sess: TensorFlow session
              states: array of states for which we want to predict the actions.
            Returns:
              The prediction of the output tensor.
            """
            prediction = sess.run(self.predictions, { self.states_: states })
            return prediction


        def update(self, sess, states, actions, targets):
            """
            Updates the weights of the neural network, based on its targets, its
            predictions, its loss and its optimizer.
            
            Args:
              sess: TensorFlow session.
              states: [current_state] or states of batch
              actions: [current_action] or actions of batch
              targets: [current_target] or targets of batch
            """
            feed_dict = { self.states_: states, self.targets_: targets, self.actions_: actions}
            _, loss = sess.run([self.train_op, self.loss], feed_dict)
            return loss
        
        
        
    


class CNNTargetNetwork(CNN):
    def __init__(self, state_dim, num_actions, history_length=3, hidden=256, lr=1e-3, tau=0.01):
        super(CNNTargetNetwork, self).__init__(state_dim, num_actions, history_length, hidden, lr)
        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):
        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        op_holder = []
        for idx,var in enumerate(tf_vars[0:total_vars//2]):
            op_holder.append(tf_vars[idx+total_vars//2].assign(
              (var.value()*self.tau) + ((1-self.tau)*tf_vars[idx+total_vars//2].value())))
        return op_holder
      
    def update(self, sess):
        for op in self._associate:
            sess.run(op)
    
