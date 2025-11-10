import os
import config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
tfd = tfp.distributions




class PolicyNetwork(keras.Model):
    def __init__(self, n_actions=2, name='actor'):
        super(PolicyNetwork, self).__init__()
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_file = './TSCmodel_aft_SCRT/REINFORCE/' + self.model_name + '.weights.h5'

        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(128, activation='relu')
        self.logits = Dense(n_actions)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.logits(x)
        logits = tf.clip_by_value(x, -10, 10)
        return logits


class Agent:
    def __init__(self, gamma=0.99, n_actions=2, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions

        self.actor = PolicyNetwork(n_actions=self.n_actions, name='actor')
        self.actor.compile(optimizer=Adam(learning_rate=config.actor_lr))


    def load_models(self,state):
        print('... loading models ...')
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        self.actor(state)
        self.actor.load_weights(self.actor.checkpoint_file)


    def choose_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        logits = self.actor(state)
        dist = tfd.Categorical(logits=logits)
        action = dist.sample()
        action = int(action.numpy()[0])
        return action