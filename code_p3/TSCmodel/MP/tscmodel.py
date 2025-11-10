import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



class Agent:
    def __init__(self):
        pass

    def choose_action(self,df_attributes):
        x_1_max = round(132 / 7)
        x_2_max = round(179 / 7)
        x_3_max = round(188 / 7)
        x_4_max = round(105 / 7)

        veh_p1_n_in = len(df_attributes[
            (df_attributes[r'Lane\Link'] == "1") &
            (df_attributes[r'Lane\Index'].isin(["2", "3"]))
            ])

        veh_p1_s_in = len(df_attributes[
            (df_attributes[r'Lane\Link'] == "5") &
            (df_attributes[r'Lane\Index'].isin(["3", "4"]))
            ])

        veh_p2_n_in = len(df_attributes[
            (df_attributes[r'Lane\Link'] == "1") &
            (df_attributes[r'Lane\Index'].isin(["1"]))
            ])

        veh_p2_s_in = len(df_attributes[
            (df_attributes[r'Lane\Link'] == "5") &
            (df_attributes[r'Lane\Index'].isin(["1", "2"]))
            ])

        veh_p3_e_in = len(df_attributes[
            (df_attributes[r'Lane\Link'] == "3") &
            (df_attributes[r'Lane\Index'].isin(["1","2","3"]))
            ])

        veh_p3_w_in = len(df_attributes[
            df_attributes[r'Lane\Link'] == "7"
            ])

        veh_p4_e_in = len(df_attributes[
            df_attributes[r'Lane\Link'].isin(["9", "10003"])
            ])

        veh_p4_w_in = len(df_attributes[
            df_attributes[r'Lane\Link'].isin(["10", "10008"])
            ])

        veh_a1_out = len(df_attributes[
            df_attributes[r'Lane\Link'].isin(["8", "11", "10002"])
            ])

        veh_a2_out = len(df_attributes[
            df_attributes[r'Lane\Link'].isin(["2"])
            ])

        veh_a3_out = len(df_attributes[
            df_attributes[r'Lane\Link'].isin(["4"])
            ])

        veh_a4_out = len(df_attributes[
            df_attributes[r'Lane\Link'].isin(["6"])
            ])

        pre_p1 = (veh_p1_n_in/(x_1_max*2)-veh_a2_out/(x_2_max*3))+(veh_p1_n_in/(x_1_max*2)-veh_a3_out/(x_3_max*2))+(veh_p1_s_in/(x_3_max*2)-veh_a1_out/x_1_max)+(veh_p1_s_in/(x_3_max*2)-veh_a4_out/(x_4_max*3))

        pre_p2 = (veh_p2_n_in/x_1_max-veh_a4_out/(x_4_max*3))+(veh_p2_s_in/(x_3_max*2)-veh_a2_out/(x_2_max*3))

        pre_p3 = (veh_p3_e_in/(x_2_max*3)-veh_a3_out/(x_3_max*2))+(veh_p3_e_in/(x_2_max*3)-veh_a4_out/(x_4_max*3)) + (veh_p3_w_in/(x_4_max*3)-veh_a2_out/(x_2_max*2))+(veh_p3_w_in/(x_4_max*3)-veh_a1_out/x_1_max)

        pre_p4 = (veh_p4_e_in/x_2_max-veh_a1_out/x_1_max)+(veh_p4_w_in/x_4_max-veh_a3_out/(x_3_max*2))

        my_list = [pre_p1,pre_p2,pre_p3,pre_p4]
        max_index = my_list.index(max(my_list))
        return max_index+1