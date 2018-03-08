# coding=utf-8

import numpy as np
import logging

from gobang_env import GobangEnv

logger = logging.getLogger('Gobang')

class GobangGame(object):
  def __init__(self, env):
    self.env = env
  
  def _reset(self):
    self.env.reset()

  def self_play(self, player):
    self._reset()
    all_states = []
    all_action_probs = []
    all_players = []
    while True:
      action, action_probs = player.get_action(self.env, return_probs=True)
      all_states.append(self.env.net_input)
      all_action_probs.append(action_probs)
      all_players.append(self.env.current_player)
      new_state, reward, done, winner = self.env.step(action)
      if done:
        all_rewards = np.zeros(len(all_players))
        all_players = np.array(all_players)
        all_rewards[all_players == winner] = 1.0
        all_rewards[all_players != winner] = -1.0
        return winner, zip(all_states, all_action_probs, all_rewards)
  
  def play(self, player1, player2):
    self._reset()
    players = {self.env.players[0]:player1,\
      self.env.players[1]:player2}
    
    while True:
      player = players[self.env.current_player]
      action = player.get_action(self.env, return_probs=False)
      state, reward, done, winner = self.env.step(action)
      if done:
        logger.debug('Winner: {}'.format(winner))
        # logger.debug('Final result: {}'.format(self.env.state.state))
        return winner
      