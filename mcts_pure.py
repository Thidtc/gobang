# coding=utf-8

'''
Pure MCTS
'''

from copy import deepcopy
import numpy as np

from utils import softmax

def uniform_policy_value(env):
  actions = list(env.available_actions)
  probs = np.ones(len(actions)) / len(actions)
  return zip(actions, probs), 0

def simulate_policy(env):
  actions = list(env.available_actions)
  action = np.random.choice(actions)
  return action

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
    select action
    '''
    action = max(self.children.items(), key=lambda node: node[1].get_value())
    return action
  
  def update(self, leaf_value):
    self.N += 1
    self.Q += 1.0 / self.N * (leaf_value - self.Q)
  
  def recursive_update(self, leaf_value):
    if self.parent is not None:
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

class MCTSPure(object):
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
    Args:
      env: the environment where simulation occurs
    '''
    node = self.root
    # Select
    done, winner = env._is_end(env.state)
    while not node.is_leaf:
      action, node = node.select()
      state, reward, done, winner = env.step(action)
    
    # Expand
    action_prob, _ = self.policy(env)
    if not done:
      node.expand(action_prob)
    leaf_value = self._evaluate_sim(env)
    # Backup
    node.recursive_update(-leaf_value)
  
  def _evaluate_sim(self, env, nlimit=1000):
    player = env.current_player
    done, winner = env._is_end(env.state)
    for i in range(nlimit):
      if done:
        break
      action = simulate_policy(env)
      new_state, reward, done, winner = env.step(action)
    if winner == -1:
      # Draw
      return 0
    else:
      return 1 if winner == player else -1
    
  
  def get_action(self, env, temp):
    '''
    Get the action together with probability
    Args:
      env: the current environment
      temp: temperature
    '''
    for i in range(self.nsim):
      self._simulate(deepcopy(env))
    # Play
    action = max(self.root.children.items(), key=lambda node:node[1].N)[0]

    return action
  
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