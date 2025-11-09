"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule, LinearModule
import cifar10_utils

import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch
    """

    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_classes == targets)

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.
    """

    total_correct = 0
    total_samples = 0

    for inputs, targets in data_loader:
        inputs_flat = inputs.reshape(inputs.shape[0], -1)
        predictions = model.forward(inputs_flat)
        pred_classes = np.argmax(predictions, axis=1)
        total_correct += np.sum(pred_classes == targets)
        total_samples += len(targets)

    avg_accuracy = total_correct / total_samples

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_dict: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    n_inputs = 32*32*3  # CIFAR10 images are 32x32x3
    n_classes = 10      # 10 classes
    # TODO: Initialize model and loss module
    model = MLP(n_inputs, hidden_dims, n_classes)
    loss_module = CrossEntropyModule()
    # TODO: Training loop including validation
    val_accuracies = []
    best_val_acc = 0.0
    best_model = None

    epoch_losses = []

    for epoch in range(epochs):
        
        batch_losses = []

        # Training
        for inputs, targets in cifar10_loader['train']:
            inputs_flat = inputs.reshape(inputs.shape[0], -1)
            
            # Forward pass 
            predictions = model.forward(inputs_flat)
            loss = loss_module.forward(predictions, targets)

            batch_losses.append(loss)

            # Backward pass
            dout = loss_module.backward(predictions, targets)
            model.backward(dout)

            # SGD update
            for layer in model.layers:
                if isinstance(layer, LinearModule):
                    layer.params['weight'] -= lr * layer.grads['weight']
                    layer.params['bias'] -= lr * layer.grads['bias']

        # Compute average epoch loss and log
        avg_epoch_loss = np.mean(batch_losses)
        epoch_losses.append(avg_epoch_loss)
          
        # Validation
        val_acc = evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = deepcopy(model)

        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])
    print(f"Test Accuracy: {test_accuracy:.4f}")
    logging_dict = {
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'test_accuracy': test_accuracy,
        "epoch_losses": epoch_losses,
    }

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(logging_dict['epoch_losses'])
    plt.title("Average Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
