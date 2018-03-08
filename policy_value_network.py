# coding = utf-8

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class Reshape(nn.Module):
  '''
  Reshape the tensor
  '''
  def __init__(self, *args):
    super(Reshape, self).__init__()
    self.shape = args

  def forward(self, x):
    return x.view(self.shape)

class PolicyValueNet(nn.Module):
  '''
  network for policy and value prediction, policy network and value network
  share part of the network
  '''
  def __init__(self, args):
    super(PolicyValueNet, self).__init__()
    self.board_width = args.board_width
    self.board_height = args.board_height
    # self.board_num = args.board_num
    self.board_num = 4
    # Shared layers of policy net and value net
    self.shared_layers = nn.Sequential(
      nn.Conv2d(self.board_num, 32, kernel_size=(3, 3), padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
      nn.ReLU(inplace=True),
    )
    # Layers of policy net
    self.policy_layers = nn.Sequential(
      nn.Conv2d(128, 4, kernel_size=(1, 1), padding=0),
      nn.ReLU(inplace=True),
      Reshape(-1, 4 * self.board_width * self.board_height),
      nn.Linear(4 * self.board_width * self.board_height, self.board_width * self.board_height),
      nn.Softmax()
    )
    # Layers of value(state value) net
    self.value_layers = nn.Sequential(
      nn.Conv2d(128, 2, kernel_size=(1, 1)),
      nn.ReLU(inplace=True),
      Reshape(-1, 2 * self.board_width * self.board_height),
      nn.Linear(2 * self.board_width * self.board_height, 64),
      nn.ReLU(inplace=True),
      nn.Linear(64, 1),
      nn.Tanh()
    )
  
  def forward(self, state):
    '''
    Args:
      state(bz, board_num, board_width, board_height): input states
    Returns:
      policy(bz, board_width * board_height): action distribution
      state value(bz, 1): state value
    '''
    z = self.shared_layers(state)
    action = self.policy_layers(z)
    value = self.value_layers(z)

    return action, value

class MLoss(nn.Module):
  def __init__(self):
    super(MLoss, self).__init__()
  def forward(self, action, value, label_action, label_value):
    # Policy loss
    policy_loss = torch.mean(torch.sum(-label_action * torch.log(action), 1))
    # Value loss
    value_loss = F.mse_loss(input=value, target=label_value)
    return policy_loss, value_loss, policy_loss + value_loss

if __name__ == '__main__':
  # TEST
  import numpy as np
  class Args(object):
    board_width = 6
    board_height = 6
    board_num = 4
  args = Args()
  net = PolicyValueNet(args)
  state = Variable(torch.from_numpy(np.zeros((10, 4, 6, 6))).float())
  policy, value = net(state)
  assert list(policy.size()) == [10, 36]
  assert list(value.size()) == [10, 1]

  net2 = MLoss()
  action = Variable(torch.from_numpy(np.zeros((10, 36))).float())
  label_action = Variable(torch.from_numpy(np.zeros((10, 36))).float())
  value = Variable(torch.from_numpy(np.zeros((10, 1))).float())
  label_value = Variable(torch.from_numpy(np.zeros((10, 1))).float())
  loss = net2(action, value, label_action, label_value)

