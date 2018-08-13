from Q import *
from util import Counter
import numpy as np


ROUND_NUM = 1

HEATMAP_SIZE = (int(np.pi*20),150)

class Qdict(Q):

    def __init__(self, discount = 0.99, alpha = 0.5):
        self.Q_dict = Counter()
        self.discount = discount
        self.alpha = alpha
        self.heatmap = np.zeros(HEATMAP_SIZE)

    def createStateVector(self, state):
        vels = state.angular_vel
        angles = state.angles
        cart_x = state.cart_x
        cart_v = state.cart_v

        state_vec = []
        #state_vec.append(round(cart_x/100,ROUND_NUM))
        for i in range(len(vels)):
            state_vec.append(round(angles[i],ROUND_NUM))
            state_vec.append(round(vels[i],ROUND_NUM))

        return tuple(state_vec)

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we never seen
          a state or (state,action) tuple
        """
        state = self.createStateVector(state)
        return self.Q_dict[(state,action)]


    def update(self, state, action, nextAction, nextState, reward, legalActions):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        state = self.createStateVector(state)
        correction = reward + self.discount*self.getQValue(nextState,nextAction)-self.Q_dict[(state,action)]
        self.Q_dict[(state,action)] += self.alpha*correction
        self.update_heatmap(state,self.Q_dict[(state,action)])


    def getMaxQValue(self,state,legalActions):
        values = []
        for a in legalActions:
            values.append(self.getQValue(state, a))
        return max(values)

    def update_heatmap(self,state,qval):
        ang = int(round(state[-2]*10,0))#%(HEATMAP_SIZE[0]-1)
        vel = int(round(state[-1]*10+HEATMAP_SIZE[1]/2,0))#%(HEATMAP_SIZE[1]-1)
        if(vel >= 0 and vel < HEATMAP_SIZE[1]):
            self.heatmap[ang, vel] = qval


