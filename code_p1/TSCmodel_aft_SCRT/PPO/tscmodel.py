import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np





class CriticNetwork(keras.Model):
    def __init__(self, name='critic'):
        super(CriticNetwork, self).__init__()
        self.model_name = name
        self.checkpoint_file = './TSCmodel_aft_SCRT/PPO/' + self.model_name + '.weights.h5'

        self.dense1 = Dense(512, activation='relu')
        self.out = Dense(1)

    def call(self, state):
        x = self.dense1(state)
        return self.out(x)


class ActorNetwork(keras.Model):
    def __init__(self, n_actions=2, name='actor'):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_file = './TSCmodel_aft_SCRT/PPO/' + self.model_name + '.weights.h5'

        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(n_actions, activation='softmax')

    def call(self, state):
        oup = self.dense1(state)
        oup = self.dense2(oup)
        return oup


class Agent:
    def __init__(self, actor_lr=0.001, critic_lr=0.001, gamma=0.99, lambd=0.1, clip_ratio=0.2, n_actions=2,
                 batch_size=64,actor_name='actor',critic_name='critic'):
        self.gamma = gamma
        self.lambd = lambd
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.clip_ratio = clip_ratio

        self.actor = ActorNetwork(n_actions=n_actions, name=actor_name)
        self.critic = CriticNetwork(name=critic_name)

        self.actor.compile(optimizer=Adam(learning_rate=actor_lr,clipnorm=0.5))
        self.critic.compile(optimizer=Adam(learning_rate=critic_lr,clipnorm=0.5))


    def load_models(self,state):
        print('... loading models ...')
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        self.actor(state)
        self.actor.load_weights(self.actor.checkpoint_file)


    def epsilon_decay(self, initial_epsilon, final_epsilon, current_episode, start_episode, end_episode):
        decay_rate = -np.log(final_epsilon / initial_epsilon) / (end_episode - start_episode)
        epsilon = initial_epsilon * np.exp(-decay_rate * (current_episode - start_episode))
        return max(final_epsilon, epsilon)


    def choose_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.actor(state)
        action = np.random.choice(self.n_actions, p=probs.numpy()[0])  # Use NumPy
        return action, probs.numpy()
