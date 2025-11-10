import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



class Agent:
    def __init__(self):
        self.t1 = 19
        self.t2 = 10
        self.t3 = 21
        self.t4 = 10

    def choose_action(self, phase):
        if phase == 1:
            action = self.t1
        elif phase == 2:
            action = self.t2
        elif phase == 3:
            action = self.t3
        elif phase == 4:
            action = self.t4
        return action
