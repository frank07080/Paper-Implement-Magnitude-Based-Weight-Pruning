# Magnitude-Based-Weight-Pruning
Magnitude-Based Weight Pruning Method with GPU with half of the VGG on CIFAR10 dataset.

The method is based on the paper "Learning both Weights and Connections for Efficient Neural Networks" written by Song Han, Jeff Pool, and etc.

This is a PyTorch implementation on this method. Instead of working with all VGG11, we only work with a half of them. Based on the paper's idea, local pruning, we fix all the linear layers and only prune convolutional layers. It is essentially the same idea and implementation if we want to fix all the conv layers and prune linear layers.

Instead of iterate over the process of pruning and retraining, we do pruning once and iterate retraining.

Result: We pruned a total of X parameters with a loss of accuracy of X%
