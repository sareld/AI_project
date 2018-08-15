from Q import *
from Qdict import *
from Qlinear import *
import random, util


class Qlearner():
    """
      Q-Learning Agent

      Functions you should fill in:
        - getQValue
        - getAction
        - getValue
        - getPolicy
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions
          for a state
    """

    def __init__(self, discount, alpha, epsilon):

        self.epsilon = epsilon
        self.myQ = Qdict(discount, alpha)
        # self.myQ = Qlinear(1)

    def getLegalActions(self):
        return []

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we never seen
          a state or (state,action) tuple
        """
        return self.myQ.getQValue(state, action)

    def getValue(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions()
        valuse = []
        if len(actions) == 0:
            return 0.0

        for action in actions:
            valuse.append(self.myQ.getQValue(state, action))

        return max(valuse)

    def getSoftMaxPolicy(self, state):
        actions = self.getLegalActions()

        if len(actions) == 0:
            return None

        m_v = -float('inf')
        max_actions = []
        values = []
        for action in actions:
            values.append(self.getQValue(state, action))

        values = np.array(values)
        values = np.power(np.e, values) / np.sum(np.power(np.e, values))

        i = np.random.choice(np.arange(0, len(actions)), p=values)
        return actions[i]

    def getPolicy(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = self.getLegalActions()

        if len(actions) == 0:
            return None

        m_v = -float('inf')
        max_actions = []
        for action in actions:
            value = self.getQValue(state, action)
            if value > m_v:
                max_actions = [action]
                m_v = value
            elif value == m_v:
                max_actions.append(action)
        if len(max_actions) > 1:
            # print("sum maxs")
            pass
        return random.choice(max_actions)

    def getSoftMaxAction(self, state):
        legalActions = self.getLegalActions()


        if len(legalActions) == 0:
            return None

        action = self.getSoftMaxPolicy(state)
        return action

    def getAction(self, state, useEpsilon=True):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action

        legalActions = self.getLegalActions()

        if len(legalActions) == 0:
            return None
        if useEpsilon and util.flipCoin(self.epsilon):
            return random.choice(legalActions)

        action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """

        self.myQ.update(state, action, self.getAction(nextState, False), nextState, reward, self.getLegalActions())
