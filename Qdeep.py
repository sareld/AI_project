from collections import deque

import tensorflow as tf
import util
import numpy as np
import random
# from Cart import CART_ACTIONS

EMPTY_VAL = 0

NUM_OF_LAYERS = 2
LAYER_DEPTH = 5
INPUT_SIZE = 2
OUTPUT_SIZE = 3

RANDOM_WEIGHTS = 0.05
RANDOM_BIASES = 0
LEARNING_RATE = 0.001

BATCH_SIZE = 100

GAMMA = 0.95
EPSILON_DECAY = 0.8
HISTORY_SIZE = 200


RIGHT = 100
LEFT = -100
STAY = 0

CART_ACTIONS = [LEFT,RIGHT,STAY]

class Qdeep():

    def __init__(self):
        self.history = deque(maxlen=HISTORY_SIZE)  # to save the history of states, actions and rewards during the game
        self.learning_rate = tf.placeholder(dtype=tf.float64, name='learning_rate')

        self.build_net()


    def weight_variable(self, name, shape):
        """Initialize weight variable
        Parameter
        ---------
        shape : list
          The shape of the initialized value.
        Returns
        -------
        The created `tf.get_variable` for weights.
        """
        initial_value = tf.random_uniform(shape=shape, minval=-0.5, maxval=0.5, dtype="float64")
        return tf.get_variable(name=name, initializer=initial_value)


    def bias_variable(self, name, shape):
        """Initialize bias variable
        Parameter
        ---------
        shape : list
          The shape of the initialized value.
        Returns
        -------
        The created `tf.get_variable` for biases.
        """
        initial_value = tf.constant([0.1], shape=shape, dtype="float64")
        return tf.get_variable(name=name, initializer=initial_value)


    def take(self, indices):
        """
        Function gets indices of the output vector and returns a mask of their values
        """
        mask = tf.one_hot(indices=indices, depth=OUTPUT_SIZE, dtype=tf.bool, on_value=True, off_value=False, axis=-1)
        x = tf.boolean_mask(self.outputs, mask)

        return x

    def build_net(self):
        with tf.name_scope('input'):
            self.x_input = tf.placeholder(dtype=tf.float64, shape=[None, INPUT_SIZE], name='x_input')
            self.action = tf.placeholder(dtype=tf.int64, shape=[None], name="action")
            self.q_est = tf.placeholder(dtype=tf.float64, shape=[None], name="q_estimation")

        first_hidden_layer = {'weights': self.weight_variable('h1_w_layer', [INPUT_SIZE, LAYER_DEPTH]),
                              'biases': self.bias_variable('h1_b_layer', [LAYER_DEPTH])}

        second_hidden_layer = {'weights': self.weight_variable('h2_w_layer', [LAYER_DEPTH,
                                                                              LAYER_DEPTH]),
                               'biases': self.bias_variable('h2_b_layer', [LAYER_DEPTH])}

        self.output_layer = {'weights': self.weight_variable('output_w_layer', [LAYER_DEPTH, OUTPUT_SIZE]),
                             'biases': self.bias_variable('output_b_layer', [OUTPUT_SIZE])}

        first_layer = tf.matmul(self.x_input, first_hidden_layer['weights']) + first_hidden_layer['biases']
        first_layer = tf.sigmoid(first_layer)

        second_layer = tf.matmul(first_layer, second_hidden_layer['weights']) + second_hidden_layer['biases']
        second_layer = tf.sigmoid(second_layer)
        output_layer = tf.matmul(second_layer, self.output_layer['weights']) + self.output_layer['biases']
        self.outputs = output_layer
        self.sess = tf.Session()



        self.loss = tf.reduce_mean(tf.square(self.q_est[0] - self.take(self.action)[0]))
        print("q =", self.q_est)
        print("take =", self.take(self.action))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.train_op = self.optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())


    def learn(self, prev_state, prev_action, reward, new_state):
        """
        the function for learning and improving the policy. it accepts the
        state-action-reward needed to learn from the final move of the game,
        and from that (and other state-action-rewards saved previously) it
        may improve the policy.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
                          This is the final state of the round.
        :param too_slow: true if the game didn't get an action in time from the
                        policy. use this to make your computation time smaller
                        by lowering the batch size for example...
        """
        # saving the sample to history
        if prev_state is not None:
            self.history.append((prev_state, prev_action, reward, new_state, 1))

        batch_size = np.min((BATCH_SIZE, len(self.history)))
        batch = random.sample(self.history, batch_size)

        prev_states = []
        prev_actions = []
        qs = []

        for i in range(batch_size):
            prev_state, prev_action, reward, new_state, done = batch[i]
            prev_states.append(np.array([prev_state.angles, prev_state.angular_vel]).flatten())
            prev_actions.append(prev_action)

            fd_v = {self.x_input: np.array([new_state.angles, new_state.angular_vel]).flatten().reshape([1, INPUT_SIZE])}
            v = np.max(self.sess.run(self.outputs, feed_dict=fd_v))
            q = reward + GAMMA * v
            qs.append(q)

        fd = {}
        fd[self.x_input] = prev_states
        fd[self.action] = prev_actions
        fd[self.q_est] = qs

        self.sess.run(self.train_op, feed_dict=fd)


    def get_action(self, state):
        """
        Function gets a state and returns its the action that maximizes the Q function (from the legal actions)
        """
        fd = {self.x_input: np.array([state.angles, state.angular_vel]).flatten().reshape([1, INPUT_SIZE])}
        net_out = self.sess.run(self.outputs, feed_dict=fd)
        return CART_ACTIONS[np.argmax(net_out)]


    def save_to_history(self, prev_state, prev_action, reward, new_state, done):
        """
        Function saves the sample to history
        """
        self.history.append((prev_state, prev_action, reward, new_state, done))


    def act(self, prev_state, prev_action, reward, new_state):
        """
        the function for choosing an action, given current state.
        it accepts the state-action-reward needed to learn from the previous
        move (which it can save in a data structure for future learning), and
        accepts the new state from which it needs to decide how to act.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
        :param too_slow: true if the game didn't get an action in time from the
                policy. use this to make your computation time smaller
                by lowering the batch size for example...
        :return: an action (from Policy.Actions) in response to the new_state.
        """
        new_rep = new_state
        prev_rep = prev_state

        # saving the sample to history
        if prev_state is not None:
            self.save_to_history(prev_rep, prev_action, reward, new_rep, 0)
        # e-greedy in train
        return self.get_action(new_state)

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we never seen
          a state or (state,action) tuple
        """
        # state = np.insert(state, 0, [1])

        fd = {self.x_input: np.array([state.angles, state.angular_vel]).flatten().reshape([1, INPUT_SIZE])}
        net_out = self.sess.run(self.outputs, feed_dict=fd)
        return net_out[0][CART_ACTIONS.index(action)]

    def update(self, state, action, nextAction, nextState, reward, legal_moves):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # self.act(state,action,reward,nextState)
        # state = np.insert(state, 0, [1])
        # nextState = np.insert(nextState,0,[1])
        self.learn(state, action, reward, nextState)
