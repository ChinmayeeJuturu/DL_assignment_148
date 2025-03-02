# DL_assignment_148
Feedforward Neural Network for MNIST Classification:

.This repository contains an implementation of a flexible feedforward neural network for classifying handwritten digits using the MNIST dataset. The model is designed to allow easy modification of hyperparameters and supports multiple optimization algorithms.

Features:

.Load the MNIST dataset using torchvision.datasets.MNIST

.Customizable number of hidden layers and neurons per layer

.Support for multiple optimization techniques:

     .SGD

      .Momentum-based gradient descent

      .Nesterov accelerated gradient descent

      .RMSprop

      .Adam

      .Nadam

.Configurable batch sizes, activation functions, and weight initialization

.Automatic train-validation-test split (90%-10% for training/validation)

.Performance evaluation using accuracy and confusion matrix

Installation:

Requirements

.Ensure you have Python installed, along with the required dependencies:
   .pip install torch torchvision matplotlib numpy
Usage:

Training the Model

Run the following command to train and evaluate the model:
python train.py
.The script will run experiments with different configurations and print the best model parameters.

Configuration:

The script supports the following hyperparameters:

  Number of epochs: [5, 10]

  Number of hidden layers: [3, 4, 5]

  Neurons per hidden layer: [32, 64, 128]

  Weight decay (L2 regularization): [0, 0.0005, 0.5]

  Learning rate: [1e-3, 1e-4]

   Optimizer: [sgd, momentum, nesterov, rmsprop, adam, nadam]

  Batch size: [16, 32, 64]

  Activation functions: [ReLU, Sigmoid]

Evaluating the Model:

.The best trained model's accuracy is evaluated on the test dataset. The confusion matrix is plotted to visualize classification performance.

Results & Recommendations:

.The best-performing hyperparameter combination will be displayed after running experiments.

.The model is evaluated using cross-entropy loss and squared error loss.

.The three best configurations for MNIST will be reported with test accuracies.

File Structure:
   .train.py       
   .README.md       
   .data/     
License:

.This project is open-source and can be modified as needed.
