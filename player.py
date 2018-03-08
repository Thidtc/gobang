# coding=utf-8

import numpy as np

from mcts import MCTS
from mcts_pure import MCTSPure, uniform_policy_value

class Player(object):
  def get_action(self, state):
    raise NotImplementedError()

class MCTSPlayer(Player):
  def __init__(self, policy_value_func, nsim, coe, temp=1e-3, is_selfplay=False):
    super(MCTSPlayer, self).__init__()
    self.mcts = MCTS(policy_value_func, coe, nsim)
    self.is_selfplay = is_selfplay
    self.temp = temp
  
  def reset(self):
    self.mcts.update(-1)
  
  def get_action(self, env, return_probs=False):
    '''
    Get an action
    Args:
      env: current environment
      temp: temperature
      return_probs: whether to return probabilites
    '''
    action_probs = np.zeros(env.board_height * env.board_width)
    if len(env.available_actions) >= 0:
      actions, probs = self.mcts.get_action_probs(env, self.temp)
      action_probs[list(actions)] = probs
      if self.is_selfplay:
        # Add some noise to encourage exploration
        action_probs_noise = 0.75 * probs + (1 - 0.75) * np.random.dirichlet(0.3 * np.ones(len(probs)))
        action = np.random.choice(actions, p=action_probs_noise)
        self.mcts.update(action)
      else:
        action = np.random.choice(actions, p=probs)
        self.mcts.update(-1)
      if return_probs:
        return action, action_probs
      else:
        return action
    return None

class MCTSPurePlayer(Player):
  def __init__(self, policy_value_func, nsim, coe, temp=1e-3, is_selfplay=False):
    super(MCTSPurePlayer, self).__init__()
    self.mcts = MCTSPure(uniform_policy_value, coe, nsim)
    self.is_selfplay = is_selfplay
    self.temp = temp
  
  def reset(self):
    self.mcts.update(-1)

  def get_action(self, env, return_probs=False):
    '''
    Get an action
    Args:
      env: current environment
      return_probs: whether to return probabilities
    '''
    if len(env.available_actions) >= 0:
      action = self.mcts.get_action(env, self.temp)
      self.mcts.update(-1)
      return action
    return None

