from abc import abstractmethod
from collections import defaultdict
import math
import numpy as np
from keras.models import Model
from mindspore import Tensor
from mindspore import Model
from mindspore.train.summary.summary_record import _get_summary_tensor_data

from evaluation.NC import NeuronCoverage


class CoverageMetrics:

    def __init__(self, model,  batch_size= None):
        self._model =  model
        self.batch_size =  batch_size
        self._activate_table = defaultdict(list)


    def _init_neuron_activate_table(self, data):
        """
        Initialise the activate table of each neuron in the model

        Args:
            data (numpy.ndarray): Data used for initialising the activate table.

        Return:
            dict, return a activate_table.
        """
        self._model.predict(Tensor(data))
        layer_out = _get_summary_tensor_data()
        activate_table = defaultdict()
        for layer, value in layer_out.items():
            activate_table[layer] = np.zeros(value.shape[1], np.bool)
        return activate_table

    def _get_bounds(self, train_dataset):
        """
        Update the lower and upper boundaries of neurons' outputs.

        Args:
            train_dataset (numpy.ndarray): Training dataset used for determine the neurons' output boundaries.

        Return:
            - numpy.ndarray, upper bounds of neuron' outputs.

            - numpy.ndarray, lower bounds of neuron' outputs.
        """
        upper_bounds = defaultdict(list)
        lower_bounds = defaultdict(list)
        batches = math.ceil(train_dataset.shape[0] / self.batch_size)
        for i in range(batches):
            inputs = train_dataset[i * self.batch_size: (i + 1) * self.batch_size]
            self._model.predict(Tensor(inputs))
            layer_out = _get_summary_tensor_data()
            for layer, tensor in layer_out.items():
                value = tensor.asnumpy()
                value = np.mean(value, axis=tuple([i for i in range(2, len(value.shape))]))
                min_value = np.min(value, axis=0)
                max_value = np.max(value, axis=0)
                if np.any(upper_bounds[layer]):
                    max_flag = upper_bounds[layer] > max_value
                    min_flag = lower_bounds[layer] < min_value
                    upper_bounds[layer] = upper_bounds[layer] * max_flag + max_value * (1 - max_flag)
                    lower_bounds[layer] = lower_bounds[layer] * min_flag + min_value * (1 - min_flag)
                else:
                    upper_bounds[layer] = max_value
                    lower_bounds[layer] = min_value
        return upper_bounds, lower_bounds

    def _activate_rate(self):
        """
        Calculate the activate rate of neurons.
        """
        total_neurons = 0
        activated_neurons = 0
        for _, value in self._activate_table.items():
            activated_neurons += np.sum(value)
            total_neurons += len(value)
        activate_rate = activated_neurons / total_neurons

        return activate_rate
class SNAC(CoverageMetrics):
    """
    Get the metric of 'super neuron activation coverage'. :math:`SNAC = |UpperCornerNeuron|/|N|`. SNAC refers to the
    proportion of neurons whose neurons output value in the test set exceeds the upper bounds of the corresponding
    neurons output value in the training set.
    """
    def __init__(self, model, train_dataset, incremental=False, batch_size=32):
        super(NeuronCoverage, self).__init__(self, model, batch_size)
        train_dataset = train_dataset
        self.upper_bounds, self.lower_bounds = self._get_bounds(train_dataset=train_dataset)


    def get_metrics(self, dataset):
        """
        Get the metric of 'strong neuron activation coverage'.
        """


        dataset =  dataset
        if not self.incremental or not self._activate_table:
            self._activate_table = self._init_neuron_activate_table(dataset[0:1])
        batches = math.ceil(dataset.shape[0] / self.batch_size)

        for i in range(batches):
            inputs = dataset[i * self.batch_size: (i + 1) * self.batch_size]
            self._model.predict(Tensor(inputs))
            layer_out = _get_summary_tensor_data()
            for layer, tensor in layer_out.items():
                value = tensor.asnumpy()
                if len(value.shape) > 2:
                    value = np.mean(value, axis=tuple([i for i in range(2, len(value.shape))]))
                activate = np.sum(value > self.upper_bounds[layer], axis=0) > 0
                self._activate_table[layer] = np.logical_or(self._activate_table[layer], activate)
        snac = self._activate_rate()
        return snac