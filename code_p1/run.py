import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from datetime import datetime
import win32com.client
import config
from vissimModel import Vissim_Server
from rl_agent import model_Agent
import random
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')




# select a TSC model for testing
TSCmodel = 'A2C'
# select mode form ["bef SCRT","aft SCRT"]
Mode = "bef SCRT"
# Mode = "aft SCRT"

config.TSCmodel = TSCmodel
if Mode == "bef SCRT":
    if TSCmodel=='DQN':
        from TSCmodel.DQN.tscmodel import Agent
    elif TSCmodel=='DDQN':
        from TSCmodel.DDQN.tscmodel import Agent
    elif TSCmodel=='A2C':
        from TSCmodel.A2C.tscmodel import Agent
    elif TSCmodel=='IQN':
        from TSCmodel.IQN.tscmodel import Agent
    elif TSCmodel=='PPO':
        from TSCmodel.PPO.tscmodel import Agent
    elif TSCmodel=='REINFORCE':
        from TSCmodel.REINFORCE.tscmodel import Agent
    elif TSCmodel=='SAC':
        from TSCmodel.SAC.tscmodel import Agent
elif Mode == "aft SCRT":
    if TSCmodel=='DQN':
        from TSCmodel_aft_SCRT.DQN.tscmodel import Agent
    elif TSCmodel=='DDQN':
        from TSCmodel_aft_SCRT.DDQN.tscmodel import Agent
    elif TSCmodel=='A2C':
        from TSCmodel_aft_SCRT.A2C.tscmodel import Agent
    elif TSCmodel=='IQN':
        from TSCmodel_aft_SCRT.IQN.tscmodel import Agent
    elif TSCmodel=='PPO':
        from TSCmodel_aft_SCRT.PPO.tscmodel import Agent
    elif TSCmodel=='REINFORCE':
        from TSCmodel_aft_SCRT.REINFORCE.tscmodel import Agent
    elif TSCmodel=='SAC':
        from TSCmodel_aft_SCRT.SAC.tscmodel import Agent

start = datetime.now()
seed = 218
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)


Vissim = win32com.client.gencache.EnsureDispatch("Vissim.Vissim.2022")
cur_dic = os.getcwd()
cur_dic = cur_dic.replace('\\', '/')
Filename_inpx = cur_dic + "/Vissim_model/Intersection_1.inpx"
Filename_layx = cur_dic + "/Vissim_model/Intersection_1.layx"
Vissim.LoadNet(Filename_inpx, False)
Vissim.LoadLayout(Filename_layx)
config.dfLinkInfo = pd.read_csv(config.pathlinkFile)

for simRun in Vissim.Net.SimulationRuns:
    Vissim.Net.SimulationRuns.RemoveSimulationRun(simRun)

agent = model_Agent(gamma=config.gamma,n_actions=config.n_action)
if TSCmodel in ["A2C","DDQN","DQN","IQN","PPO","REINFORCE"]:
    test_agent = Agent(n_actions=39)
elif TSCmodel in ["SAC"]:
    test_agent = Agent(n_actions=1,state_dim=1147)

config.action_slice = np.arange(7, 46, 1)
dummy_state = np.random.random((1, 1147))
if TSCmodel in ["IQN"]:
    test_agent.load_models(dummy_state, config.iqn_N_eval)
else:
    test_agent.load_models(dummy_state)

config.test_agent = test_agent

cumulative_reward_history = []
best_avg_reward = -float('inf')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def map(reward):
    x = 2*(reward-0)/(8-0)-1
    t = sigmoid(x*5)
    t = t*5
    return reward*t

state_demand = [config.total_demand]
state = np.array(state_demand)
CurPhase = 1
done = True
reward_li = []
for episode in range(1,config.n_episodes+1):
    remainder = (episode) % (config.episode_reset_interval)
    if remainder == 0 and episode != (config.n_episodes):
        Vissim.Exit()
        time.sleep(5)
        Vissim = win32com.client.gencache.EnsureDispatch("Vissim.Vissim.2022")
        cur_dic = os.getcwd()
        cur_dic = cur_dic.replace('\\', '/')
        Filename_inpx = cur_dic + "/Vissim_model/Intersection_1.inpx"
        Filename_layx = cur_dic + "/Vissim_model/Intersection_1.layx"
        Vissim.LoadNet(Filename_inpx, False)
        Vissim.LoadLayout(Filename_layx)

        # delete all the previous simulation runs
        for simRun in Vissim.Net.SimulationRuns:
            Vissim.Net.SimulationRuns.RemoveSimulationRun(simRun)

    VissimCOM = Vissim_Server(Vissim)
    action1,actions = agent.choose_action(state)
    actions = tf.convert_to_tensor(actions)
    actions = tf.reshape(actions, (1, 12))

    real_action1 = tf.round(action1 * state_demand[0])

    num_veh = VissimCOM.ActionExecution(real_action1, CurPhase)
    Vissim.Simulation.Stop()
    VissimCOM.Reset_Vissim()
    safety = VissimCOM.GetReward()
    if safety!=-10000:
        likelihood = VissimCOM.GetActionLikelihood(action1)
        reward_ = (safety/num_veh)*likelihood
        reward = reward_*config.beta
        reward = map(reward)

        if episode>200:
            agent.actor.optimizer.learning_rate.assign(0.001)

        agent.learn(state,actions,reward)
        cumulative_reward_history.append(reward)
        avg_rewards = np.mean(cumulative_reward_history[-30:])

        if reward > best_avg_reward:
            best_avg_reward = reward
            agent.save_models()

        print('episode ', episode, 'reward %.3f, average reward %.3f' % (reward, avg_rewards))
        with open('episode_logs.txt', 'a') as file:
            file.write('Episode {}, Safety {:.1f}, Likelihood {:.12f}, Reward {:.3f}, Average Reward {:.3f}\n'.format(episode, safety, likelihood, reward, avg_rewards))
            file.write('Action {}\n'.format(action1))

Vissim.Exit()
cumulative_reward_history = np.array(cumulative_reward_history)
np.save(config.pathSavedCumRewardFiles+"_"+TSCmodel+".npy", cumulative_reward_history)

duration = (datetime.now()-start).total_seconds()/3600
print('Total computational time : %s h' %duration)
