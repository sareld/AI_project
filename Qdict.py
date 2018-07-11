from Q import *
from util import Counter
import pandas as pd

class Qdict(Q):

    def __init__(self, discount = 0.8, alpha = 0.5 ):
        self.Q_dict = Counter()
        self.discount = discount
        self.alpha = alpha


    def discreteSate(self,state):
        pends = []
        for pend in state:
            pends.append((round(pend[0],2),round(pend[1],2)))

        return tuple(pends)

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we never seen
          a state or (state,action) tuple
        """
        state = self.discreteSate(state)
        return self.Q_dict[(state,action)]

    def update(self, state, action, nextAction, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        state = self.discreteSate(state)

        correction = reward + self.discount*self.getQValue(nextState,nextAction)-self.Q_dict[(state,action)]
        self.Q_dict[(state,action)] += self.alpha*correction