import numpy as np
from itertools import product
import numpy as np
import time
import os

from multiprocessing import Pool
from tqdm import tqdm
from keras.models import load_model, Model
from scipy.stats import gaussian_kde

from utils import *
from keras_tuner.engine.logger_test import build_model
from tensorflow.python.ops import nn


class NeuralNetwork(nn.Module):
  def __init__(self, input_size, num_layers, num_units, activation_fn):
    super(NeuralNetwork, self).__init__()
    self.layers = nn.ModuleList()
    self.layers.append(nn.Linear(input_size, num_units))
    for _ in range(num_layers - 1):
      self.layers.append(nn.Linear(num_units, num_units))
    self.activation_fn = activation_fn

  def forward(self, x):
    for layer in self.layers:
        x = self.activation_fn(layer(x))
    return x


def generate_solution(learning_rate, batch_size, num_epochs, regularization_param, num_layers, num_units, activation_fn, weight_init_method, optimizer):
    # Generate the solution based on the given hyperparameters
    solution = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'regularization_param': regularization_param,
        'num_layers': num_layers,
        'num_units': num_units,
        'activation_fn': activation_fn,
        'weight_init_method': weight_init_method,
        'optimizer': optimizer
    }

    return solution

def latin_hypercube_sampling(num_samples, param_ranges):
    num_params = len(param_ranges)
    samples = []

    for _ in range(num_samples):
        sample = []
        for param_range in param_ranges:
            sample.append(np.random.choice(param_range))
        samples.append(sample)

    return samples



def evaluate_model(hyperparameters):
    # 根据超参数配置训练和评估模型
    learning_rate = hyperparameters['learning_rate']
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']
    regularization_param = hyperparameters['regularization_param']
    num_layers = hyperparameters['num_layers']
    num_units = hyperparameters['num_units']
    activation_fn = hyperparameters['activation_fn']
    weight_init_method = hyperparameters['weight_init_method']
    optimizer = hyperparameters['optimizer']

    # 构建模型并进行训练
    model = build_model(learning_rate, batch_size, num_epochs, regularization_param,
                        num_layers, num_units, activation_fn, weight_init_method, optimizer)
    model.train()

    # 评估模型并计算适应度度量


    return model
