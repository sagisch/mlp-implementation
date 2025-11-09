"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm=False):
        """
        Initializes MLP object.
        Implement module setup of the network.
        The linear layer have to initialized according to the Kaiming initialization.
        Add the Batch-Normalization _only_ is use_batch_norm is True.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ELU layer.
        """

        super().__init__()

        self.layers = nn.ModuleList()
        layer_dims = [n_inputs] + n_hidden + [n_classes]

        for layer in range(len(n_hidden)):
            
            linear_layer = nn.Linear(layer_dims[layer], layer_dims[layer+1])

            nn.init.kaiming_normal_(linear_layer.weight, nonlinearity='relu')
            nn.init.zeros_(linear_layer.bias)

            self.layers.append(linear_layer)

            if use_batch_norm:
              self.layers.append(nn.BatchNorm1d(layer_dims[layer+1]))

            self.layers.append(nn.ELU())
        
        linear_layer = nn.Linear(layer_dims[-2], layer_dims[-1])

        nn.init.kaiming_normal_(linear_layer.weight, nonlinearity='relu')
        nn.init.zeros_(linear_layer.bias)

        self.layers.append(linear_layer)

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """

        for layer in self.layers:
            x = layer(x)
        out = x 

        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
    
