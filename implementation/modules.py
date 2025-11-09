"""
This module implements various modules of the network.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        # Initializing weight parameters using Kaiming initialization
        std = np.sqrt(2/in_features) # Kaiming N(0, sqrt(2/N))
        self.params['weight'] = np.random.randn(out_features, in_features) * std 

        # Initializing biases with zeros
        self.params['bias'] = np.zeros(out_features)

        # Initializing gradients with zeros
        self.grads['weight'] = np.zeros((out_features, in_features)) # size N*M (out*in)
        self.grads['bias'] = np.zeros(out_features)


    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """

        self.x = x
        # x is size S*M, W is size N*M
        out = x @ self.params['weight'].T + self.params['bias']

        return out


    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """

        self.grads['weight'] = dout.T @ self.x
        self.grads['bias'] = np.sum(dout, axis=0)
        dx = dout @ self.params['weight']

        return dx


    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """

        self.x = None


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self, alpha):
        self.alpha = alpha


    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """

        out = np.where(x >= 0, x, self.alpha*(np.exp(x)-1))
        self.x = x 

        return out


    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """

        dx = dout * np.where(self.x >= 0, 1, self.alpha*np.exp(self.x))

        return dx


    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """

        self.x = None


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """

        x_shifted = x - np.max(x, axis=1, keepdims=True)
        ex = np.exp(x_shifted)
        out = ex / np.sum(ex, axis=1, keepdims=True)
        self.out = out

        return out


    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """

        s = self.out
        dx = s * (dout - np.sum(dout * s, axis=1, keepdims=True))

        return dx


    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """
        self.out = None


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
        """

        batch_size = x.shape[0]
        out = -np.sum(np.log(x[np.arange(batch_size), y])) / batch_size

        return out


    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
        """

        batch_size = x.shape[0]
        # dx = -y/ x / batch_size
        dx = np.zeros_like(x)
        dx[np.arange(batch_size), y] = -1 / (x[np.arange(batch_size), y] * batch_size)

        return dx