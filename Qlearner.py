from Q import *
from Qdeep import *
from Qdict import *
import random, util


class ReinforcementAgent:
  """
    Abstract Reinforcemnt Agent: A ValueEstimationAgent
	  which estimates Q-Values (as well as policies) from experience
	  rather than a model

      What you need to know:
		  - The environment will call
		    observeTransition(state,action,nextState,deltaReward),
		    which will call update(state, action, nextState, deltaReward)
		    which you should override.
      - Use self.getLegalActions(state) to know which actions
		    are available in a state
  """
  ####################################
  #    Override These Functions      #
  ####################################

  def update(self, state, action, nextState, reward):
    """
	    This class will call this function, which you write, after
	    observing a transition and reward
    """
    pass

  ####################################
  #    Read These Functions          #
  ####################################

  def observeTransition(self, state,action,nextState,deltaReward):
    """
    	Called by environment to inform agent that a transition has
    	been observed. This will result in a call to self.update
    	on the same arguments

    	NOTE: Do *not* override or call this function
    """
    self.episodeRewards += deltaReward
    self.update(state,action,nextState,deltaReward)

  def startEpisode(self):
    """
      Called by environment when new episode is starting
    """
    self.lastState = None
    self.lastAction = None
    self.episodeRewards = 0.0

  def stopEpisode(self):
    """
      Called by environment when episode is done
    """
    if self.episodesSoFar < self.numTraining:
      self.accumTrainRewards += self.episodeRewards
    else:
      self.accumTestRewards += self.episodeRewards
    self.episodesSoFar += 1
    if self.episodesSoFar >= self.numTraining:
      # Take off the training wheels
      self.epsilon = 0.0    # no exploration
      self.alpha = 0.0      # no learning

  def isInTraining(self):
      return self.episodesSoFar < self.numTraining

  def isInTesting(self):
      return not self.isInTraining()

  def __init__(self, actionFn = None, numTraining=100, epsilon=0.0, alpha=0.9, gamma=1):
    """
    actionFn: Function which takes a state and returns the list of legal actions

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    self.actionFn = actionFn
    self.episodesSoFar = 0
    self.accumTrainRewards = 0.0
    self.accumTestRewards = 0.0
    self.numTraining = int(numTraining)
    self.epsilon = float(epsilon)
    self.alpha = float(alpha)
    self.discount = float(gamma)



class Qlearner(ReinforcementAgent):
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
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.myQ = Qdeep()
        self.sign = 1


    def getLegalActions(self):
        return []


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we never seen
          a state or (state,action) tuple
        """
        return self.myQ.getQValue(state,action)


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
            valuse.append(self.myQ.getQValue(state,action))

        return max(valuse)


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
            value = self.getQValue(state,action)
            if value > m_v:
                max_actions = [action]
                m_v = value
            elif value == m_v:
                max_actions.append(action)

        return random.choice(max_actions)

    def getAction(self, state):
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
        #TODO: delete this line to implement none random action choice
        #return random.choice(legalActions)


        if len(legalActions) == 0:
            return None
        if util.flipCoin(self.epsilon):
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
        self.myQ.update(state,action,self.getAction(nextState), nextState, reward)
