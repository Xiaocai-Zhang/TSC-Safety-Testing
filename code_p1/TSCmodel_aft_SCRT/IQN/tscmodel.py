import os
import config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.layers import LeakyReLU




class IQNNetwork(keras.Model):
    def __init__(self, n_actions=2, d_model=128, n_cos=64, name='actor'):
        super(IQNNetwork, self).__init__(name=name)
        self.n_actions = n_actions
        self.d_model = d_model
        self.n_cos = n_cos
        self.checkpoint_file = './TSCmodel_aft_SCRT/IQN/' + name + '.weights.h5'

        self.fc1 = Dense(1024, kernel_initializer='he_normal')
        self.act1 = LeakyReLU(alpha=0.01)
        self.fc2 = Dense(d_model, kernel_initializer='he_normal')
        self.act2 = LeakyReLU(alpha=0.01)

        self.tau_fc = Dense(d_model, kernel_initializer='he_normal')
        self.tau_act = LeakyReLU(alpha=0.01)

        self.head1 = Dense(128, kernel_initializer='he_normal')
        self.head_act = LeakyReLU(alpha=0.01)
        self.head2 = Dense(n_actions, kernel_initializer='he_normal')

    def call(self, state, taus):
        B = tf.shape(state)[0]
        N = tf.shape(taus)[1]

        # ψ(s): [B, d_model]
        x = self.fc1(state)
        x = self.act1(x)
        x = self.fc2(x)
        psi_s = self.act2(x)                          # [B, d_model]
        psi_s = tf.expand_dims(psi_s, axis=1)         # [B, 1, d_model]
        psi_s = tf.repeat(psi_s, repeats=N, axis=1)   # [B, N, d_model]

        # φ(τ) with cosine basis
        i = tf.range(1, self.n_cos + 1, dtype=tf.float32)[tf.newaxis, tf.newaxis, :]  # [1,1,n_cos]
        cos_embed = tf.cos(np.pi * i * taus)          # [B, N, n_cos]
        phi = self.tau_fc(cos_embed)                  # [B, N, d_model]
        phi = self.tau_act(phi)

        h = psi_s * phi                                # [B, N, d_model]
        h = self.head1(h)
        h = self.head_act(h)
        quantiles = self.head2(h)                      # [B, N, n_actions]
        return quantiles


class Agent:
    def __init__(self, input_dims, alpha=0.0005, gamma=0.99, n_actions=2, tau=0.005, batch_size=64,
                 N=32, N_dash=32, N_eval=32, n_cos=64, d_model=128, actor_name='actor'):
        self.gamma = gamma
        self.tau_polyak = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.input_dims = input_dims

        self.N = N
        self.N_dash = N_dash
        self.N_eval = N_eval
        self.kappa = 1.0  # Huber κ

        # Networks
        self.actor = IQNNetwork(n_actions=n_actions, d_model=d_model, n_cos=n_cos, name=actor_name)
        self.actor_target = IQNNetwork(n_actions=n_actions, d_model=d_model, n_cos=n_cos, name='target_' + actor_name)

        self.actor.compile(optimizer=Adam(learning_rate=alpha, clipnorm=0.5))
        self.actor_target.compile(optimizer=Adam(learning_rate=alpha, clipnorm=0.5))

        self._hard_update_target_network()

    @staticmethod
    def _sample_taus(batch_size, N):
        return tf.random.uniform((batch_size, N, 1), minval=0., maxval=1., dtype=tf.float32)

    def _hard_update_target_network(self):
        for (t, s) in zip(self.actor_target.trainable_variables, self.actor.trainable_variables):
            t.assign(s)


    def load_models(self, state, iqn_N_eval):
        print('... loading models ...')
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        taus = self._sample_taus(1, iqn_N_eval)
        self.actor(state, taus)
        self.actor.load_weights(self.actor.checkpoint_file)

    @staticmethod
    def epsilon_decay(initial_epsilon, final_epsilon, current_episode, start_episode, end_episode):
        decay_rate = -np.log(final_epsilon / initial_epsilon) / (end_episode - start_episode)
        epsilon = initial_epsilon * np.exp(-decay_rate * (current_episode - start_episode))
        return max(final_epsilon, epsilon)

    def choose_action(self, state, episode, test=False):
        epsilon = self.epsilon_decay(1.0, 0.1, episode, 1, config.n_episodes)
        state = tf.convert_to_tensor(state[None,:], dtype=tf.float32)
        taus = self._sample_taus(1, self.N_eval)
        quantiles = self.actor(state, taus)
        q_expectation = tf.reduce_mean(quantiles, axis=1)
        if not test and np.random.random() < epsilon:
            action = int(np.random.randint(self.n_actions))
        else:
            action = int(tf.argmax(q_expectation, axis=1).numpy()[0])
        return action


    def random_action(self):
        return int(tf.random.uniform((1,), minval=0, maxval=self.n_actions, dtype=tf.int32).numpy()[0])
