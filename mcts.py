# coding=utf-8

'''
MCTS in alphago zero
'''

from copy import deepcopy
import numpy as np

from utils import softmax

class Node(object):
  def __init__(self, parent, prob, coe):
    '''
    Args:
      parent: parent node
      prior: prior probability
      coe: coefficient of U
    '''
    self.parent = parent
    self.children = {}
    self.N = 0
    self.Q = 0
    self.U = 0
    self.P = prob
    self.V = 0
    self.coe = coe
  
  def expand(self, priors):
    '''
    Expand the tree node according to the prior probability
    Args:
      priors: a list of (action, probability) pair, produced by policy network
    '''
    for action, prob in priors:
      if action not in self.children:
        self.children[action] = Node(self, prob, self.coe)
  
  def select(self):
    '''
    Select action
    '''
    action = max(self.children.items(), key=lambda node: node[1].get_value())
    return action
  
  def update(self, leaf_value):
    self.N += 1
    self.Q += 1.0 / self.N * (leaf_value - self.Q)
  
  def recursive_update(self, leaf_value):
    if self.parent is not None:
      #self.parent.recursive_update(self.V)
      self.parent.recursive_update(-leaf_value)
    self.update(leaf_value)
  
  def get_value(self):
    # self.U = self.P / (1 + self.N)
    self.U = self.P * np.sqrt(self.parent.N) / (1 + self.N)
    return self.Q + self.coe * self.U

  @property
  def is_leaf(self):
    return self.children == {}

  @property
  def is_root(self):
    return self.parent is None

class MCTS(object):
  def __init__(self, policy, coe, nsim):
    '''
    Args:
      policy: the policy
      coe: coefficient of U
      nsim: number of simulations      
    '''
    self.policy = policy
    self.coe = coe
    self.nsim = nsim
    self.root = Node(None, 1.0, self.coe)

  def _simulate(self, env):
    '''
    Perform one simulation
    '''
    node = self.root
    # Select
    done, winner = env._is_end(env.state)
    while not node.is_leaf:
      action, node = node.select()
      env.state, reward, done, winner = env.step(action)
    # Expand
    action_prob, leaf_value = self.policy(env)
    if not done:
      node.expand(action_prob)
    else:
      if winner == -1:
        leaf_value = 0
      else:
        leaf_value = 1.0 if winner == env.current_player else -1.0
    # Backup
    node.recursive_update(-leaf_value)

  def get_action_probs(self, env, temp):
    for i in range(self.nsim):
      self._simulate(deepcopy(env))
    # Play
    action_N = [(action, node.N) for action, node in self.root.children.items()]
    actions, Ns = zip(*action_N)
    # This line may cause Warning because Ns may contain zero
    probs = softmax(1.0 / temp * np.log(Ns))

    return actions, probs

  def update(self, action):
    '''
    Move the root of the tree to the children according to the action
    Args:
      action: the performed action
    '''
    if action in self.root.children:
      self.root = self.root.children[action]
      self.root.parent = None
    else:
      self.root = Node(None, 1.0, self.coe)
