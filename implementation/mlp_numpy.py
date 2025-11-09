"""
This module implements a multi-layer perceptron (MLP) in NumPy.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
        """

        self.layers = []
        layer_sizes = [n_inputs] + n_hidden + [n_classes]
        self.num_layers = len(layer_sizes)

        for i in range(len(layer_sizes)-1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            self.layers.append(LinearModule(in_features=in_dim, out_features=out_dim))
            if i < self.num_layers - 2:
              self.layers.append(ELUModule(alpha=1.0))

        self.layers.append(SoftMaxModule())            


    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """

        out = x
        for layer in self.layers:
            out = layer.forward(out)

        return out


    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss
        """

        grad = dout
        for layer in reversed(self.layers):
            grad = layer.backward(grad)


    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.
        """
        
        for layer in self.layers:
            if hasattr(layer, 'clear_cache'):
                layer.clear_cache()
