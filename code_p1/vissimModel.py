import time
import config
import pandas as pd
pd.set_option('chained_assignment',None)
import tensorflow as tf
import SSAM_auto as SSAM
import numpy as np
from joblib import load




# demand occurrence probability estimation
kmeans_loaded = load(config.kmeansModelPath)
prob_df = pd.read_csv(config.kmeansProbabilitiesFile)


class Vissim_Server:
    def __init__(self, Vissim):
        self.Vissim = Vissim
        self.Step = 0

    def InferNextPhase(self, curentPhases):
        nxtPhase = config.nextPhasesRule[curentPhases]
        return nxtPhase

    def InferLastPhase(self, curentPhases):
        lastPhase = config.lastPhasesRule[curentPhases]
        return lastPhase

    def InitParameters(self,demands):
        ''' Simulation Parameters '''
        self.Vissim.Simulation.SetAttValue('SimPeriod', config.simulation_period)
        Sim_rdsd = 101
        self.Vissim.Simulation.SetAttValue('RandSeed', Sim_rdsd)
        self.Vissim.Simulation.SetAttValue('SimRes', config.simulation_resolution)
        if config.graphics == 0:
            self.Vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 1)
            self.Vissim.SuspendUpdateGUI()

        # dummy demand for warming-up
        tensor = tf.constant([[0,2,3,2],[3,0,2,3],[3,3,0,3],[2,2,2,0]], dtype=tf.float32)
        demands = [tensor]+demands
        n = 0
        for tensor in demands:
            n+=1
            for row_index in range(4):
                for col_index in range(4):
                    demand_value = tensor[row_index, col_index].numpy()
                    self.Vissim.Net.Matrices.ItemByKey(n).SetValue(row_index + 1, col_index + 1, demand_value)
        return None

    def InitVehicleComposition(self):
        # car, SUV, HGV, Bus, Van, Lite Truck
        Rel_flow = [0.45,0.35,0.05,0.05,0.05,0.05]
        Speed = 60
        len_Veh_compo = len(config.VehTypelist)
        Attributes = ("VehType", "DesSpeedDistr", "RelFlow")

        values = ()
        for i in range(len_Veh_compo):
            buff = (config.VehTypelist[i], Speed, Rel_flow[i])
            values = values + (buff,)
        self.Vissim.Net.VehicleCompositions.ItemByKey(1).VehCompRelFlows.SetMultipleAttributes(Attributes, values)
        return None


    def ConverttoODMatrix(self,array):
        array = tf.squeeze(array, axis=0)
        matrix = tf.zeros((4, 4), dtype=tf.float32)
        mask = tf.ones((4, 4), dtype=tf.float32) - tf.eye(4, dtype=tf.float32)
        indices = tf.where(mask == 1)
        matrix = tf.tensor_scatter_nd_update(matrix, indices, array)
        return matrix


    def Reset_Vissim(self):
        for simRun in self.Vissim.Net.SimulationRuns:
            self.Vissim.Net.SimulationRuns.RemoveSimulationRun(simRun)
        time.sleep(3)
        return None


    def Run_Sim_Warmup(self, duration):
        for a in range(duration):
            self.Vissim.Simulation.RunSingleStep()
            self.Step += 1
        return None


    def Run_Sim(self, duration):
        for a in range(int(duration)):
            self.Vissim.Simulation.RunSingleStep()
            self.Step += 1
        return None


    def SetFirstPhases(self):
        # First set all the phases to RED
        for SG in [1,2,3,4]:
            self.Vissim.Net.SignalControllers.ItemByKey(1).SGs.ItemByKey(SG).SetAttValue('SigState','RED')
        return None


    def GetVehNum(self):
        all_veh_attributes = self.Vissim.Net.Vehicles.GetMultipleAttributes(('Lane\Link', 'Lane\Index', 'No', 'Speed', 'Pos', 'VehType'))
        df_attributes = pd.DataFrame(all_veh_attributes,columns=['Lane\Link', 'Lane\Index', 'No', 'Speed', 'Pos', 'VehType'])
        df_attributes = df_attributes[df_attributes['VehType'].isin(['100', '200', '300', '630', '640', '650'])]
        df_attributes = df_attributes.dropna(subset=['Pos'])
        veh_num_li = df_attributes['No'].unique().tolist()
        return veh_num_li


    def ScDurationExecution(self,CurPhase,time):
        lastPhase = self.InferLastPhase(CurPhase)
        veh_num_li_all = []
        for i in range(3):
            self.Vissim.Net.SignalControllers.ItemByKey(1).SGs.ItemByKey(lastPhase).SetAttValue('SigState', 'AMBER')
            veh_num_li = self.GetVehNum()
            veh_num_li_all = veh_num_li_all+veh_num_li
            self.Run_Sim(1)
        for i in range(2):
            self.Vissim.Net.SignalControllers.ItemByKey(1).SGs.ItemByKey(lastPhase).SetAttValue('SigState', 'RED')
            veh_num_li = self.GetVehNum()
            veh_num_li_all = veh_num_li_all + veh_num_li
            self.Run_Sim(1)
        for i in range(time):
            self.Vissim.Net.SignalControllers.ItemByKey(1).SGs.ItemByKey(CurPhase).SetAttValue('SigState','GREEN')
            veh_num_li = self.GetVehNum()
            veh_num_li_all = veh_num_li_all + veh_num_li
            self.Run_Sim(1)
        return veh_num_li_all


    def EncodePhase(self, phase):
        code = [1, 2, 3, 4]
        res = [0, 0, 0, 0]
        indx = code.index(phase)
        res[indx] = 1
        return np.array(res)


    def GetTrafficSignalOup(self,CurPhase):
        PhasesCode = self.EncodePhase(CurPhase)
        state_traffic = self.GetTrafficState()
        state = np.concatenate((state_traffic, PhasesCode))
        if config.TSCmodel in ['DDQN','DQN','IQN']:
            action = config.test_agent.choose_action(state, episode=1, test=True)
            real_action = config.action_slice[action]
        elif config.TSCmodel in ['A2C','REINFORCE']:
            action = config.test_agent.choose_action(state)
            real_action = config.action_slice[action]
        elif config.TSCmodel in ['SAC']:
            action = config.test_agent.choose_action(state, test=True)
            real_action = round(0.5 * (action[0] + 1) * (46 - 7) + 7)
        elif config.TSCmodel in ['PPO']:
            action,_ = config.test_agent.choose_action(state)
            real_action = config.action_slice[action]
        return real_action

    def ActionExecution(self, action1, CurPhase):
        demand1 = self.ConverttoODMatrix(action1)
        # set demand matrix
        self.InitParameters([demand1])
        # set vehicle composition
        self.InitVehicleComposition()

        self.Run_Sim_Warmup(config.simulation_warm_up)
        self.SetFirstPhases()

        veh_num_li = []
        while True:
            Signal_t = self.GetTrafficSignalOup(CurPhase)
            veh_num = self.ScDurationExecution(CurPhase,Signal_t)
            veh_num_li = veh_num_li+veh_num
            nxtPhase = self.InferNextPhase(CurPhase)
            CurPhase = nxtPhase
            if self.Step >= config.simualtion_execution:
                break

        unique_num = set(veh_num_li)
        num_veh = len(unique_num)
        return num_veh

    def CalReward(self,ttc):
        w = self.Sigmoid(ttc)
        r = w * (5-ttc)
        return r

    def Sigmoid(self,x):
        k = 1
        x0 = 2.5
        return 1 / (1 + np.exp(k * (x - x0)))


    def GetReward(self):
        vissimFolder = './Vissim_model/Vissim_output'
        try:
            SSAM.Start(vissimFolder)
            results = pd.read_csv(vissimFolder+'/Intersection_1_001.csv')
            results['reward'] = results.apply(lambda row: self.CalReward(row['TTC']),axis=1)
            overall_reward = results['reward'].sum()
            return overall_reward
        except:
            return -10000


    def DivideLengthIntoSegments(self, total_length, segment_length):
        full_segments = total_length // segment_length
        remaining_segment = total_length % segment_length
        number_of_segments = int(full_segments + (1 if remaining_segment > 0 else 0))
        rangList = []
        for i in range(number_of_segments+1):
            rang = [i * segment_length, (i + 1) * segment_length]
            rangList.append(rang)
        return rangList

    def FindSegIndex(self,row,rangList):
        pos = float(row['Pos'])
        for i in range(len(rangList)):
            rang = rangList[i]
            if pos>=rang[0] and pos<rang[1]:
                return i

    def GetFeatures(self, row, df_veh):
        link = row['link']
        lane = row['lane']
        length = float(row['length'])
        df_veh_lane = df_veh[(df_veh['Lane\Link'] == str(int(link))) & (df_veh['Lane\Index'] == float(lane))]
        rangList = self.DivideLengthIntoSegments(length, config.linkSegment)
        res = np.zeros((len(rangList), 3))
        if df_veh_lane.empty == False:
            df_veh_lane['seg_index'] = df_veh_lane.apply(lambda row: self.FindSegIndex(row, rangList), axis=1, result_type='expand')
            for index, row in df_veh_lane.iterrows():
                try:
                    segIndex = int(row['seg_index'])
                    res[segIndex, 0] = 1
                    res[segIndex, 1] = row['Speed']
                    res[segIndex, 2] = row['Acceleration']
                except:
                    pass
        res = self.StateNormalization(res)
        res = res.reshape(-1)
        return res

    def ExtractFeatures(self,df_attributes,dfLinkInfo):
        df_veh = df_attributes[df_attributes['VehType'].isin(['100', '200','300','630','640','650'])]
        featureLi = []
        for index, row in dfLinkInfo.iterrows():
            feature = self.GetFeatures(row, df_veh)
            featureLi.append(feature)
        TsFeature = np.concatenate(featureLi, axis=0)
        return TsFeature

    def GetTrafficState(self):
        all_veh_attributes = self.Vissim.Net.Vehicles.GetMultipleAttributes(('Lane\Link','Lane\Index','No','Speed','Acceleration ','Pos','VehType'))
        df_attributes = pd.DataFrame(all_veh_attributes,columns=['Lane\Link','Lane\Index','No','Speed','Acceleration','Pos','VehType'])
        state_traffic = self.ExtractFeatures(df_attributes,config.dfLinkInfo)
        return state_traffic

    def StateNormalization(self,state_traffic):
        state_traffic[:,0] = 2*state_traffic[:,0]-1
        state_traffic[:,1] = 2*(state_traffic[:,1]/config.maxSpeed)-1
        state_traffic[:,2] = 2*((state_traffic[:,2]-config.minAcceleration)/(config.maxAcceleration-config.minAcceleration))-1
        return state_traffic

    def GetActionLikelihood(self,action1):
        a1 = action1[0].numpy()
        a1 = a1.astype(np.float64)
        a1 = a1.reshape(1, 12)
        label1 = kmeans_loaded.predict(a1)
        prob_df_a1 = prob_df[prob_df['class']==int(label1[0])]
        prob_a1 = prob_df_a1['probability'].tolist()[0]
        return prob_a1
