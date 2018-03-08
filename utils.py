# coding=utf-8

import numpy as np

def softmax(xs):
  '''
  Args:
    xs: numpy array
  '''
  xs = np.exp(xs - np.max(xs))
  xs /= np.sum(xs)
  return xs
  