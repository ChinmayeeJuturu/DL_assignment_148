# DL_assignment_148
1.Feedforward Neural Network for MNIST Classification:

.This repository contains an implementation of a flexible feedforward neural network for classifying handwritten digits using the MNIST dataset. The model is designed to allow easy modification of hyperparameters and supports multiple optimization algorithms.

2.Features:

.Load the MNIST dataset using torchvision.datasets.MNIST

.Customizable number of hidden layers and neurons per layer

.Support for multiple optimization techniques:
     1.SGD
     2.Momentum-based gradient descent
     3.Nesterov accelerated gradient descent
     4.RMSprop
     5.Adam
     5.Nadam

.Configurable batch sizes, activation functions, and weight initialization

.Automatic train-validation-test split (90%-10% for training/validation)

.Performance evaluation using accuracy and confusion matrix

3.Installation:
..Requirements

.Ensure you have Python installed, along with the required dependencies:
   pip install torch torchvision matplotlib numpy
4.Usage:

..Training the Model

.The script will run experiments with different configurations and print the best model parameters.

5.Configuration:

.The script supports the following hyperparameters:

  Number of epochs: [5, 10]

  Number of hidden layers: [3, 4, 5]

  Neurons per hidden layer: [32, 64, 128]

  Weight decay (L2 regularization): [0, 0.0005, 0.5]

  Learning rate: [1e-3, 1e-4]

   Optimizer: [sgd, momentum, nesterov, rmsprop, adam, nadam]

  Batch size: [16, 32, 64]

  Activation functions: [ReLU, Sigmoid]

6.Evaluating the Model:

.The best trained model's accuracy is evaluated on the test dataset. The confusion matrix is plotted to visualize classification performance.

7.Results & Recommendations:

.The best-performing hyperparameter combination will be displayed after running experiments.

.The model is evaluated using cross-entropy loss and squared error loss.

.The three best configurations for MNIST will be reported with test accuracies.
    
8.License:

.This project is open-source and can be modified as needed.

9.Best Model Configuration:
Model Architecture
Hidden Layers: 3 layers with 128 neurons each.

Activation Function: ReLU (Rectified Linear Unit).

Output Layer: Softmax (for classification tasks).

Training Configuration:
Batch Size: 16

Optimizer: Nadam

Loss Function: Categorical Cross-Entropy (for classification tasks).

Epochs: 5

Performance:
Training Accuracy: 0.9775 (at Epoch 4)

Validation Accuracy: 0.9748 (at Epoch 4)

Training Loss: 178.5357 (at Epoch 4)

