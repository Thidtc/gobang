# coding=utf-8

import argparse
import torch
import numpy as np
from torch import optim
from torch.autograd import Variable
from collections import deque, Counter
import random
import logging
import sys

from player import MCTSPlayer, MCTSPurePlayer
from mcts_pure import uniform_policy_value
from policy_value_network import PolicyValueNet, MLoss
from gobang_env import GobangEnv
from gobang_game import GobangGame

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Gobang")
  parser.add_argument('--board_width', type=int, default=6, help="Width of the board")
  parser.add_argument('--board_height', type=int , default=6, help="Height of the board")
  parser.add_argument('--n_in_row', type=int, default=4, help="Number of pieces in a row")
  # parser.add_argument('--board_num', type=int, default=int, help="Number of input filter")
  parser.add_argument('--t', type=float, default=1., help="Temperature")
  parser.add_argument('--coe', type=float, default=5., help="Coefficient thats balance exploration")
  parser.add_argument('--nrun', type=int, default=1500, help="Game runs")
  parser.add_argument('--nrun_eval', type=int , default=10, help="Game runs for a single evaluation")
  parser.add_argument('--bz', type=int, default=512, help="Batch size")
  parser.add_argument('--rb_size', type=int, default=10000, help="Replay buffer size")
  parser.add_argument('--policy_update_interval', type=int, default=5, help="Interval for policy update")
  parser.add_argument('--lr', type=float, default=5e-3, help="Learning rate")
  parser.add_argument('--model_file', type=str, default="model.pt")
  parser.add_argument('--nsample_run', type=int, default=1, help="Number of sample procedure in one iteration")
  parser.add_argument('--nsim', type=int, default=400, help="Number of simulations in MCTS")
  parser.add_argument('--nsim_pure', type=int, default=1000, help="Number of simulation for pure MCTS")
  parser.add_argument('--eval_interval', type=int, default=50)
  parser.add_argument('--gpu', type=int, default=0, help="GPU number")
  parser.add_argument('--weight_decay', type=float, default=1e-4, help="Optimizer weight decay")
  parser.add_argument('--k', type=int, default=5, help="k steps per update")
  parser.add_argument('--target_kl', type=float, default=0.025, help="target KL divergence")
  parser.add_argument('--temp', type=float, default=1e-3, help="Softmax temperature")

  args = parser.parse_args()

  # Init logger
  logger = logging.getLogger('Gobang')
  formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.formatter = formatter
  logger.addHandler(console_handler)
  logger.setLevel(logging.DEBUG)

  # Cuda device initialization
  torch.cuda.set_device(args.gpu)

  env = GobangEnv(args)
  game = GobangGame(env)
  net = PolicyValueNet(args).cuda()
  loss_func = MLoss().cuda()
  args.lr_multiplier = 1.
  optimizer = optim.Adam(params=net.parameters(), lr=args.lr_multiplier * args.lr, weight_decay=args.weight_decay)

  # The policy based on NN
  def net_policy_value(env):
    x = env.net_input
    x = Variable(torch.from_numpy(x).float()).cuda()
    x = x.view([1] + list(x.size()))
    # return net(x.unsqueeze(0))
    policy, value = net(x)
    policy = policy.data.cpu().numpy().squeeze()
    value = value.data.cpu().numpy().squeeze()
    policy_probs = zip(range(policy.shape[0]), policy)
    # Filter valid moves
    policy_probs = zip(env.available_actions, policy[env.available_actions])
    return policy_probs, value
  mcts_player = MCTSPlayer(net_policy_value, args.nsim, args.coe,\
    temp=args.temp, is_selfplay=True)

  # Replay buffer
  rb = deque(maxlen=args.rb_size)

  def argument_data(data):
    '''
    Argument the data by rotation and flip
    Args:
      data: a list of (state, probs, reward)
    Returns:
      new_data: a list of argmented data
    '''
    new_data = []
    for state, probs, reward in data:
      for i in [1, 2, 3, 4]:
        # Rotate
        new_state = np.array([np.rot90(s, i) for s in state])
        new_probs = np.rot90(probs.reshape(args.board_height, args.board_width), i)
        new_data.append((new_state, new_probs.flatten(), reward))

        # Flip
        new_state = np.array([np.fliplr(s) for s in new_state])
        new_probs = np.fliplr(new_probs)
        new_data.append((new_state, new_probs.flatten(), reward))
    return new_data

  def collect_selfplay_data(nsample_run, temp):
    '''
    Collect data via selfplay
    Args:
      nsample_run: Number of run to collect data
      temp: temperature for MCTS
    '''
    data_lens = []
    for _ in range(nsample_run):
      winner, data = game.self_play(mcts_player)
      data = list(data)
      data_lens.append(len(data))
      data = argument_data(data)
      rb.extend(data)
    return data_lens

  def update_policy():
    '''
    Update the policy
    '''
    batch_data = random.sample(rb, args.bz)
    states = np.array([item[0] for item in batch_data])
    probs = np.array([item[1] for item in batch_data])
    rewards = np.array([item[2] for item in batch_data])
    states = Variable(torch.from_numpy(states).float()).cuda()
    probs = Variable(torch.from_numpy(probs).float()).cuda()
    rewards = Variable(torch.from_numpy(rewards).float()).cuda()

    net.eval()
    old_action_probs, old_v = net(states)
    for i in range(args.k):
    # for i in range(1):
      # Training
      net.train()
      action_probs, value = net(states)
      policy_loss, value_loss, loss = loss_func(action_probs,\
        value, probs, rewards)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Get current result
      net.eval()
      action_probs, value = net(states)

      # Entropy
      entropy = torch.mean(torch.sum(action_probs * torch.log(action_probs + 1e-7), 1))

      # Stop the update from changing too much, the change between network is
      # calculated by KL divergence
      # KL divergence between old action probabilities and current action probabilities
      kl_divergence = torch.mean(torch.sum(old_action_probs *\
        (torch.log(old_action_probs + 1e-7) - torch.log(action_probs + 1e-7)), 1))
      kl_divergence = kl_divergence.cpu().data[0]
      if kl_divergence > args.target_kl * 4:
        break
    if kl_divergence > args.target_kl * 2 and args.lr_multiplier > 0.1:
      args.lr_multiplier /= 1.5
    elif kl_divergence < args.target_kl * 2 and args.lr_multiplier < 10:
      args.lr_multiplier *= 1.5
    optimizer.lr = args.lr_multiplier * args.lr
    logger.debug('kl_divergence: {}'.format(kl_divergence))
    logger.debug('lr_multiplier: {}'.format(args.lr_multiplier))

    return loss.data[0]
  
  def evaluate():
    '''
    Evaluate the policy
    '''
    net.eval()
    new_mcts_player = MCTSPlayer(net_policy_value, args.nsim, args.coe, temp=args.temp)
    mcts_pure_player = MCTSPurePlayer(uniform_policy_value, args.nsim_pure, args.coe, temp=args.temp)
    result = Counter()
    for i in range(args.nrun_eval):
      winner = game.play(new_mcts_player, mcts_pure_player)
      result.update([winner])
    logger.info('Win: {}, lose: {}, draw: {}'.format(result[1], result[2], result[-1]))
    winning_rate = (result[1] + 0.5 * result[-1]) / args.nrun_eval
    return winning_rate

  # Main training process
  try:
    best_win_rate = 0.
    # res = evaluate()
    # logger.info('Eval win_rate: {}'.format(res))
    for i in range(args.nrun):
      data_lens = collect_selfplay_data(args.nsample_run, args.temp)
      for data_len in data_lens:
        logger.info('Sample - trajectory len: {}'.format(data_len))

      l = 'NaN'
      if len(rb) > args.bz:
        l = update_policy()
        logger.info('Updata - run {}, loss: {}'.format(i, l))
      
      # Evaluate
      if i % args.eval_interval == args.eval_interval - 1:
        win_rate = evaluate()
        logger.info('Eval win_rate: {}'.format(win_rate))

        # Store the model
        if win_rate > best_win_rate:
          best_win_rate = win_rate
          torch.save(net.state_dict(), args.model_file)

  except Exception as e:
    pass