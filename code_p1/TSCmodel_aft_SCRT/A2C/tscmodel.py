import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import numpy as np





class ActorCriticModel(tf.keras.Model):
    def __init__(self, n_actions=2, name='actor'):
        super(ActorCriticModel, self).__init__()
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_file = './TSCmodel_aft_SCRT/A2C/' + self.model_name + '.weights.h5'

        self.common1 = Dense(512, activation='relu', kernel_initializer='he_normal')
        self.common2 = Dense(128, activation='relu', kernel_initializer='he_normal')
        self.actor = Dense(n_actions, activation='softmax')
        self.critic = Dense(1)

    def call(self, inputs):
        x = self.common1(inputs)
        x = self.common2(x)
        return self.actor(x), self.critic(x)


class Agent:
    def __init__(self, alpha=0.001, gamma=0.99, n_actions=2, tau=0.005,
                 batch_size=64,actor_name='actor'):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor_critic = ActorCriticModel(n_actions=n_actions, name=actor_name)
        self.actor_critic.compile(optimizer=RMSprop(learning_rate=alpha))


    def load_models(self,state):
        print('... loading models ...')
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        self.actor_critic(state)
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)


    def epsilon_decay(self, initial_epsilon, final_epsilon, current_episode, start_episode, end_episode):
        decay_rate = -np.log(final_epsilon / initial_epsilon) / (end_episode - start_episode)
        epsilon = initial_epsilon * np.exp(-decay_rate * (current_episode - start_episode))
        return max(final_epsilon, epsilon)


    def choose_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs,_ = self.actor_critic(state)
        action_probs = tf.squeeze(action_probs).numpy()
        action = np.random.choice(self.n_actions, p=np.squeeze(action_probs))
        return action
