import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np





LOG_STD_MIN = -20.0
LOG_STD_MAX =  2.0
EPS = 1e-6

class ActorGaussian(Model):
    def __init__(self, n_actions=2, name='actor'):
        super().__init__(name=name)
        self.checkpoint_file = f'./TSCmodel/SAC/{name}.weights.h5'
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.mu = Dense(n_actions, activation=None)
        self.log_std = Dense(n_actions, activation=None)

    def _sample(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        mu = self.mu(x)
        log_std = tf.clip_by_value(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)
        # reparameterization
        eps = tf.random.normal(shape=tf.shape(mu))
        z = mu + std * eps
        a = tf.tanh(z)
        # log_prob with tanh correction
        log_prob = -0.5 * (((z - mu) / (std + EPS))**2 + 2.0*log_std + np.log(2.0*np.pi))
        log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
        # correction: log(1 - tanh(z)^2) = log(1 - a^2)
        correction = tf.reduce_sum(tf.math.log(1.0 - tf.square(a) + EPS), axis=1, keepdims=True)
        log_prob = log_prob - correction
        return a, log_prob, mu, log_std

    def call(self, state, sample=True):
        if sample:
            a, logp, mu, log_std = self._sample(state)
            return a, logp
        else:
            x = self.fc1(state)
            x = self.fc2(x)
            mu = self.mu(x)
            return tf.tanh(mu), None


class CriticNetwork(Model):
    def __init__(self, name='critic'):
        super().__init__(name=name)
        self.checkpoint_file = f'./TSCmodel/SAC/{name}.weights.h5'
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(256, activation='relu')
        self.q_out = Dense(1, activation=None)

    def call(self, state, action):
        sa = tf.concat([state, action], axis=1)
        x = self.fc1(sa)
        x = self.fc2(x)
        q = self.q_out(x)
        return q


class Agent:
    def __init__(self, n_actions,state_dim,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005, batch_size=64,
                 target_entropy=None, reward_scale=1.0):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.reward_scale = reward_scale

        # Networks
        self.actor = ActorGaussian(n_actions, name='actor')
        self.critic_1 = CriticNetwork(name='critic_1')
        self.critic_2 = CriticNetwork(name='critic_2')
        self.target_critic_1 = CriticNetwork(name='target_critic_1')
        self.target_critic_2 = CriticNetwork(name='target_critic_2')

        # build variables
        dummy_s = tf.zeros((1, state_dim), dtype=tf.float32)
        dummy_a = tf.zeros((1, n_actions), dtype=tf.float32)
        _ = self.actor(dummy_s, sample=False)
        _ = self.critic_1(dummy_s, dummy_a)
        _ = self.critic_2(dummy_s, dummy_a)
        _ = self.target_critic_1(dummy_s, dummy_a)
        _ = self.target_critic_2(dummy_s, dummy_a)

        # Optimizers
        self.actor_opt = Adam(learning_rate=actor_lr)
        self.critic1_opt = Adam(learning_rate=critic_lr)
        self.critic2_opt = Adam(learning_rate=critic_lr)

        # Temperature (entropy) parameter alpha with auto-tuning
        if target_entropy is None:
            target_entropy = -float(n_actions)  # common default
        self.target_entropy = target_entropy
        # log_alpha as variable
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.alpha_opt = Adam(learning_rate=alpha_lr)

        # hard update target critics
        self._hard_update(self.target_critic_1, self.critic_1)
        self._hard_update(self.target_critic_2, self.critic_2)

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def _hard_update(self, target: Model, source: Model):
        for tp, sp in zip(target.trainable_variables, source.trainable_variables):
            tp.assign(sp)

    def load_models(self,state):
        self.actor.load_weights(self.actor.checkpoint_file)

    def choose_action(self, state, test=False):
        s = tf.convert_to_tensor([state], dtype=tf.float32)
        if test:
            a, _ = self.actor(s, sample=False)  # deterministic: tanh(mu)
            return a[0].numpy()
        else:
            a, _ = self.actor(s, sample=True)   # stochastic sample
            return a[0].numpy()