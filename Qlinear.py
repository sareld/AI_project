from Q import *
import numpy as np

class Qlinear(Q):


    def __init__(self, pendu_num, discount = 0.8, alpha = 0.000001):
        self.alpha = alpha
        self.discount = discount
        self.pend_num = pendu_num
        self.featurs_num = 4*pendu_num
        self.W = np.zeros((1,self.featurs_num))

    def normalAngle(self,ang):
        base = np.pi
        return ang - base * round(float(ang)/base)

    def getValue(self, state):
        return 0

    def feature(self,state,action):
        feature = np.zeros((self.featurs_num,1))
        j = 0
        for i in range(self.pend_num*6+1,6):
            feature[i+1] = self.normalAngle(state.angles[j])
            feature[i+2] = state.angular_vel[j]
            feature[i+3] = action*state.angles[j]
            feature[i+4] = action*state.angular_vel[j]
            j+=1
        return feature

    def getQValue(self, state, action):
        res = self.W.dot(self.feature(state,action))[0,0]
        return res

    def getMaxQValue(self,state,legalActions):
        values = []
        for a in legalActions:
            values.append(self.getQValue(state, a))

        return max(values)

    def update(self, state, action, nextAction, nextState, reward, legalActions):

        a = self.getMaxQValue(nextState,legalActions)

        delta = self.alpha * (reward + self.discount * self.getQValue(nextState, a) -
                              self.getQValue(state, action)) * (self.feature(state, action))

        # delta = self.alpha*(reward + self.discount*self.getQValue(nextState,nextAction) -
        #                     self.getQValue(state,action))*(self.feature(state,action))

        #print(delta)
        self.W += delta.T

