import os
import config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp




class ActorNetwork(keras.Model):
    def __init__(self, n_dim_1=128, n_dim_2=64, n_actions=1, name='actor'):
        super(ActorNetwork, self).__init__()
        self.n_dim_1 = n_dim_1
        self.n_dim_2 = n_dim_2
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_file = './save/' + self.model_name + '.weights.h5'

        self.dense1 = Dense(self.n_dim_1,activation='relu',kernel_initializer='he_normal')
        self.dense2 = Dense(self.n_dim_2,activation='relu',kernel_initializer='he_normal')
        D = 61
        self.oup1 = Dense(D)
        self.oup2 = Dense(D)
        self.oup3 = Dense(D)
        self.oup4 = Dense(D)
        self.oup5 = Dense(D)
        self.oup6 = Dense(D)
        self.oup7 = Dense(D)
        self.oup8 = Dense(D)
        self.oup9 = Dense(D)
        self.oup10 = Dense(D)
        self.oup11 = Dense(D)
        self.oup12 = Dense(D)

    def call(self, state):
        oup = self.dense1(state)
        oup = self.dense2(oup)
        logit1 = self.oup1(oup)
        logit2 = self.oup2(oup)
        logit3 = self.oup3(oup)
        logit4 = self.oup4(oup)
        logit5 = self.oup5(oup)
        logit6 = self.oup6(oup)
        logit7 = self.oup7(oup)
        logit8 = self.oup8(oup)
        logit9 = self.oup9(oup)
        logit10 = self.oup10(oup)
        logit11 = self.oup11(oup)
        logit12 = self.oup12(oup)
        return [logit1,logit2,logit3,logit4,logit5,logit6,logit7,logit8,logit9,logit10,logit11,logit12]


class model_Agent:
    def __init__(self, gamma=0.99, n_actions=2, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions

        self.actor = ActorNetwork(n_actions=n_actions, name='actor')
        self.actor.compile(optimizer=Adam(learning_rate=config.actor_lr))

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)

    def load_models(self,state):
        print('... loading models ...')
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state = self.normalization_s(state)
        self.actor(state)
        self.actor.load_weights(self.actor.checkpoint_file)

    def map_logits(self,index):
        index = tf.cast(index, tf.float32)
        res = -3+index*0.1
        return res

    def normalization_s(self,state):
        s_min = 0
        s_max = 1000
        state_ = 2*((state-s_min)/(s_max-s_min))-1
        return state_


    def choose_action(self, state):
        state = self.normalization_s(state)
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        act_values = self.actor(state)
        actions = []
        real_actions = []
        for value in act_values:
            dist = tfp.distributions.Categorical(logits=value)
            action = dist.sample()[0]
            actions.append(action)
            real_action = self.map_logits(action)
            real_actions.append(real_action)

        action1 = tf.nn.softmax(real_actions, axis=-1)
        action1 = tf.reshape(action1, (1, 12))
        action1 = tf.cast(action1, dtype=tf.float32)
        return action1,actions

    def learn(self,state,actions,reward):
        state_s = self.normalization_s(state)
        state_s = tf.convert_to_tensor([state_s], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape() as tape:
            logits_list = self.actor(state_s)
            total_log_prob = 0.0
            total_entropy = 0.0
            for i, (logit, action_taken) in enumerate(zip(logits_list, actions[0])):
                dist = tfp.distributions.Categorical(logits=logit)
                log_prob = dist.log_prob(action_taken)
                total_log_prob += log_prob
                total_entropy += dist.entropy()
            avg_entropy = total_entropy / len(logits_list)
            actor_loss = -total_log_prob * reward - 0.01 * avg_entropy
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
