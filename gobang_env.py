# coding=utf-8

import gym
import numpy as np
from copy import deepcopy
from itertools import takewhile

class BoardState(object):
  def __init__(self, board_width, board_height):
    self.board_width = board_width
    self.board_height = board_height
    self.state = np.zeros((self.board_height, self.board_width), dtype=np.int8)
    self.action = -1
    self.player = -1

class GobangEnv(gym.Env):
  '''
  Gobang environment
  for a 3 * 3 Gobang game, the action for the board is
  0 1 2
  3 4 5
  6 7 8
  '''
  def __init__(self, args):
    super(GobangEnv, self).__init__()
    self.board_width = args.board_width
    self.board_height = args.board_height
    self.n_in_row = args.n_in_row
    self.players = [1, 2]
    self.reset()

  def _reset(self):
    self.current_player = self.players[0]
    self.state = BoardState(self.board_width, self.board_height)
    self.available_actions = list(range(self.board_width * self.board_height))
  
  def _action2pos(self, action):
    x, y = action // self.board_width, action % self.board_width
    return x, y

  def _pos2action(self, pos):
    x, y = pos
    action = x * self.board_width + y
    return action

  def _step(self, action):
    assert action in self.available_actions
    x, y = self._action2pos(action)
    self.available_actions.remove(action)
    new_state = deepcopy(self.state)
    new_state.state[x][y] = self.current_player
    new_state.action = action
    new_state.player = self.current_player
    done, winner = self._is_end(new_state)
    reward = 1.0 if done else 0.0
    self.current_player = self.players[1] if self.current_player == self.players[0]\
      else self.players[0]
    self.state = new_state
    return new_state, reward, done, winner
  
  def _render(self, mode, close):
    pass
  
  def _close(self):
    pass
  
  def _seed(self, seed):
    pass

  @property
  def net_input(self):
    '''
    States for NN input
    Args:
      the current player in the game
    Returns:
      a numpy tensor
      state_res[0]: the current player's moves
      state_res[1]: the opponent player's moves
      state_res[2]: the last move
      state_res[3]: whether the current player is the first player
    '''
    state_res = np.zeros((4, self.board_width, self.board_height))
    state_res[0][self.state.state == self.state.player] = 1.0
    state_res[1][self.state.state == (self.players[0] if self.state.player == self.players[1]\
      else self.players[1])] = 1.0
    x, y = self._action2pos(self.state.action)
    # When the action is -1, that is no action, the x, y will be (board_height-1, board_width-1)
    state_res[2][x][y] = 1.0
    if (self.board_width * self.board_width - len(self.available_actions)) \
      % 2 == 0:
      state_res[3,:,:] += 1.0
    return state_res

  def _is_end(self, state):
    '''
    Is a state end
    Returns:
      is_end: Whether is the state an end state
      winner: is the state is end, return the winner(If draw, return -1), otherwise return -1
    '''
    player, action = state.player, state.action
    x, y = self._action2pos(action)

    # Game not over
    if self.board_width * self.board_height - len(self.available_actions)\
      < 2 * self.n_in_row:
      return False, -1

    def l_func(x):
      return len(list(takewhile(lambda x: state.state[x[0]][x[1]] == player, x)))
    left = [(x, yi) for yi in range(y - 1, -1, -1)]
    right = [(x, yi) for yi in range(y + 1, self.board_width)]
    cnt = 1 + l_func(left) + l_func(right)
    if cnt == self.n_in_row:
      return True, player # Game over

    left = [(xi, y) for xi in range(x - 1, -1, -1)]
    right = [(xi, y) for xi in range(x + 1, self.board_height)]
    cnt = 1 + l_func(left) + l_func(right)
    if cnt == self.n_in_row:
      return True, player

    left = [(x - i, y - i) for i in range(1, min(x, y))]
    right = [(x + i, y + i) for i in range(1, min(self.board_height - x, self.board_width - y))]
    cnt = 1 + l_func(left) + l_func(right)
    if cnt == self.n_in_row:
      return True, player

    left = [(x + i, y - i) for i in range(1, min(self.board_width - x, y + 1))]
    right = [(x - i, y + i) for i in range(1, min(x + 1, self.board_height - y))]
    cnt = 1 + l_func(left) + l_func(right)
    if cnt == self.n_in_row:
      return True, player
    
    # Draw, no more avaiblable actions
    if len(self.available_actions) == 0:
      return True, -1
    return False, -1

def TEST_env():
  def uniform_policy(env):
    actions = list(env.available_actions)
    probs = np.ones(len(actions)) / len(actions)
    action = np.random.choice(actions)
    return action, probs

  class Arg(object):
    board_width = 6
    board_height = 6
    n_in_row = 4
  args = Arg()
  env = GobangEnv(args)
  global_step = 0
  while True:
    global_step += 1
    action, action_probs = uniform_policy(env)
    new_state, reward, done, winner = env.step(action)
    print("step", global_step)
    print(env.current_player)
    print(action)
    print(env.state.state)
    if done:
      print(winner)
      print(reward)
      break

if __name__ == '__main__':
  TEST_env()
