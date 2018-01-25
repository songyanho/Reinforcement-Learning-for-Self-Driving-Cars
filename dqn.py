"""DQN Class
DQN(NIPS-2013)
"Playing Atari with Deep Reinforcement Learning"
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
DQN(Nature-2015)
"Human-level control through deep reinforcement learning"
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
"""
import numpy as np
import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_float('discount_rate', 0.99, 'Initial discount rate.')
flags.DEFINE_integer('replay_memory_length', 50000, 'Number of replay memory episode.')
flags.DEFINE_integer('target_update_count', 5, 'DQN Target Network update count.')
flags.DEFINE_integer('max_episode_count', 5000, 'Number of maximum episodes.')
flags.DEFINE_integer('batch_size', 64, 'Batch size. (Must divide evenly into the dataset sizes)')
flags.DEFINE_integer('frame_size', 1, 'Frame size. (Stack env\'s observation T-n ~ T)')
flags.DEFINE_string('model_name', 'MLPv1', 'DeepLearning Network Model name (MLPv1, ConvNetv1)')
flags.DEFINE_float('learning_rate', 0.0001, 'Batch size. (Must divide evenly into the dataset sizes)')
flags.DEFINE_string('gym_result_dir', 'gym-results/', 'Directory to put the gym results.')
flags.DEFINE_string('gym_env', 'CartPole-v0', 'Name of Open Gym\'s enviroment name. (CartPole-v0, CartPole-v1, MountainCar-v0)')
flags.DEFINE_boolean('step_verbose', False, 'verbose every step count')
flags.DEFINE_integer('step_verbose_count', 100, 'verbose step count')
flags.DEFINE_integer('save_step_count', 2000, 'model save step count')
flags.DEFINE_string('checkpoint_path', 'checkpoint/', 'model save checkpoint_path (prefix is gym_env)')


class DeepQNetwork:
    def __init__(self, session, model_name, input_size, output_size, learning_rate=0.0001, frame_size=1, name="main"):
        """DQN Agent can
        1) Build network
        2) Predict Q_value given state
        3) Train parameters
        Args:
            session (tf.Session): Tensorflow session
            input_size (int): Input dimension
            output_size (int): Number of discrete actions
            name (str, optional): TF Graph will be built under this name scope
        """
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.frame_size = frame_size

        self.net_name = name
        self.learning_rate = learning_rate

        self._build_network(model_name=model_name)

    def _build_network(self, model_name="MLPv1"):
        with tf.variable_scope(self.net_name):

            if self.frame_size > 1:
                X_shape = [None] + list(self.input_size) + [self.frame_size]
            else:
                X_shape = [None] + list(self.input_size)
            self._X = tf.placeholder(tf.float32, X_shape, name="input_x")

            models = {
                "MLPv1": MLPv1,
                "ConvNetv1": ConvNetv1,
                "ConvNetv2": ConvNetv2
            }

            model = models[model_name](self._X, self.output_size,
                                       frame_size=self.frame_size, learning_rate=self.learning_rate)
            model.build_network()

            self._Qpred = model.inference
            self._Y = model.Y
            self._loss = model.loss
            self._train = model.optimizer

    def predict(self, state):
        """Returns Q(s, a)
        Args:
            state (np.ndarray): State array, shape (n, input_dim)
        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        """

        if self.frame_size > 1:
            x_shape = [-1] + list(self.input_size) + [self.frame_size]
        else:
            x_shape = [-1] + list(self.input_size)
        x = np.reshape(state, x_shape)
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        """Performs updates on given X and y and returns a result
        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)
        Returns:
            list: First element is loss, second element is a result from train step
        """

        feed = {
            self._X: x_stack,
            self._Y: y_stack
        }
        return self.session.run([self._loss, self._train], feed)


class MLPv1:

    def __init__(self, X, num_classes, frame_size, learning_rate=0.001):
        state_length = X.get_shape().as_list()[1]
        self.X = tf.reshape(X, [-1, state_length])

        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self):
        net = self.X
        net = tf.layers.dense(net, 16, activation=tf.nn.relu)
        net = tf.layers.dense(net, 64, activation=tf.nn.relu)
        net = tf.layers.dense(net, 32, activation=tf.nn.relu)
        net = tf.layers.dense(net, self.num_classes)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)


class ConvNetv1:

    def __init__(self, X, num_classes, frame_size=1, learning_rate=0.001):
        self.X = tf.reshape(X, [-1, 128, frame_size])

        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self):
        conv1 = tf.layers.conv1d(self.X, 32, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)

        conv2 = tf.layers.conv1d(pool1, 64, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)

        conv3 = tf.layers.conv1d(pool2, 128, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2)
        pool3_flat = tf.reshape(pool3, [-1, 16 * 128])

        net = tf.layers.dense(pool3_flat, 512)
        net = tf.layers.dense(net, 128)
        net = tf.layers.dense(net, self.num_classes)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)



class ConvNetv2:

    def __init__(self, X, num_classes, frame_size=1, learning_rate=0.001):
        self.X = tf.reshape(X, [-1, 128, frame_size])

        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self):
        conv1 = tf.layers.conv1d(self.X, 128, kernel_size=7, padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=4, strides=2)

        conv2 = tf.layers.conv1d(pool1, 256, kernel_size=5, padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=3, strides=2)

        conv3 = tf.layers.conv1d(pool2, 512, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2)

        conv4 = tf.layers.conv1d(pool3, 512, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2)

        conv5 = tf.layers.conv1d(pool4, 512, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool5 = tf.layers.max_pooling1d(inputs=conv5, pool_size=2, strides=2)
        pool5_flat = tf.reshape(pool5, [-1, 3 * 512])

        net = tf.layers.dense(pool5_flat, 1024)
        net = tf.layers.dense(net, 256)
        net = tf.layers.dense(net, self.num_classes)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)