import os
import config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.layers import LeakyReLU




class ActorNetwork(keras.Model):
    def __init__(self, n_actions=2, name='actor'):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_file = './TSCmodel_aft_SCRT/DDQN/' + self.model_name + '.weights.h5'

        self.dense1 = Dense(1024,kernel_initializer='he_normal')
        self.dense2 = Dense(128,kernel_initializer='he_normal')
        self.dense3 = Dense(n_actions, kernel_initializer='he_normal')
        self.activation1 = LeakyReLU(alpha=0.01)

    def call(self, state):
        oup = self.dense1(state)
        oup = self.activation1(oup)
        oup = self.dense2(oup)
        oup = self.activation1(oup)
        oup = self.dense3(oup)
        return oup


class Agent:
    def __init__(self, alpha=0.001, gamma=0.99, n_actions=2, tau=0.005,
                 batch_size=64,actor_name='actor'):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise_std_dev = 0.2

        self.actor = ActorNetwork(n_actions=n_actions, name=actor_name)
        self.actor_target = ActorNetwork(n_actions=n_actions, name='target_'+actor_name)

        self.actor.compile(optimizer=Adam(learning_rate=alpha,clipnorm=0.5))
        self.actor_target.compile(optimizer=Adam(learning_rate=alpha,clipnorm=0.5))

        self._update_target_value_network(self.actor_target, self.actor)

    def _update_target_value_network(self, target, source):
        for (a, b) in zip(target.trainable_variables, source.trainable_variables):
            a.assign(self.tau * b + (1 - self.tau) * a)


    def load_models(self,state):
        print('... loading models ...')
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        self.actor(state)
        self.actor.load_weights(self.actor.checkpoint_file)


    def epsilon_decay(self, initial_epsilon, final_epsilon, current_episode, start_episode, end_episode):
        decay_rate = -np.log(final_epsilon / initial_epsilon) / (end_episode - start_episode)
        epsilon = initial_epsilon * np.exp(-decay_rate * (current_episode - start_episode))
        return max(final_epsilon, epsilon)


    def choose_action(self, state, episode, test=False):
        epsilon = self.epsilon_decay(1, 0.1, episode, 1, config.n_episodes)
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        act_values = self.actor(state)
        if test == False:
            if np.random.random() < epsilon:
                action = self.random_action()
            else:
                action = tf.argmax(act_values, axis=1).numpy()[0]
            return action
        else:
            action = tf.argmax(act_values, axis=1).numpy()[0]
            return action


    def random_action(self):
        action = tf.random.uniform((1,), minval=0, maxval=self.n_actions, dtype=tf.int32)
        return action