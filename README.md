# DeepLearning-Fundamentals
## Deep Neural Network Layers and Functions

This repository contains a collection of custom-built deep neural network layers, activation functions, loss functions, and a high-level neural network implementation. Additionally, it includes implementations of cross-correlation methods with various padding modes, pooling methods (including Max Pooling, Average Pooling, and Global Average Pooling). With these components, you can create and train neural networks from scratch. The code provided is designed to help you understand the inner workings of deep learning models and serve as a foundation for building more complex networks.

## Table of Contents

- [Layers](#layers)
  - [Dense Layer](#dense-layer)
  - [Activation Layer](#activation-layer)
  - [Pooling Layers](#pooling-layers)
  - [Conv2D Layer](#conv2d-layer)
  - [Dropout Layer](#dropout-layer)
  - [Flatten Layer](#flatten-layer)
  - [Global Average Pooling Layer](#global-average-pooling-layer)
- [Activation Functions](#activation-functions)
  - [Sigmoid](#sigmoid)
  - [ReLU](#relu)
  - [Tanh](#tanh)
  - [Softmax](#softmax)
- [Loss Functions](#loss-functions)
  - [Mean Absolute Error (MAE)](#mean-absolute-error-mae)
  - [Mean Squared Error (MSE)](#mean-squared-error-mse)
  - [Binary Cross-Entropy](#binary-cross-entropy)
  - [Categorical Cross-Entropy](#categorical-cross-entropy)
- [Convolution Methods](#convolution-methods)
  - [Cross-Correlation with Padding](#cross-correlation-with-padding)
  - [Pooling Methods](#pooling-methods)
- [Neural Network Implementation](#neural-network-implementation)
  - [Network Class](#network-class)
  - [Usage](#usage)
- [Example Usage](#example-usage)
  - [XOR Data](#xor-data)
  - [MNIST Data](#mnist-data)

## Layers

### Dense Layer

The `DenseLayer` is a fully connected layer with adjustable output dimensions. It includes weight and bias initialization, forward propagation, and backward propagation.

### Activation Layer

The `ActivationLayer` supports various activation functions, including Sigmoid, ReLU, Tanh, and Softmax, and their derivatives. It can be added after any layer to introduce non-linearity.

### Pooling Layers

Pooling layers include Max Pooling, Average Pooling, and Global Average Pooling. These layers are essential for down-sampling and reducing spatial dimensions in convolutional neural networks.

### Conv2D Layer

The `Conv2DLayer` implements a 2D convolutional layer with customizable kernel sizes and numbers of filters. It supports forward and backward propagation for image data.

### Dropout Layer

The `DropoutLayer` introduces dropout regularization during training to prevent overfitting.

### Flatten Layer

The `FlattenLayer` reshapes input data into a flat vector, typically used before connecting to fully connected layers.

### Global Average Pooling Layer

The `GlobalAveragePoolingLayer` computes the average of each feature map over the spatial dimensions, resulting in a global average representation for each feature.

## Activation Functions

### Sigmoid

The Sigmoid activation function squashes input values into the range (0, 1) and is commonly used in binary classification.

### ReLU

ReLU (Rectified Linear Unit) is a popular activation function that introduces non-linearity by returning zero for negative inputs and the input value for positive inputs.

### Tanh

The Tanh activation function is similar to the Sigmoid but maps inputs to the range (-1, 1).

### Softmax

Softmax is used in multi-class classification problems to convert raw scores into probability distributions over multiple classes.

## Loss Functions

### Mean Absolute Error (MAE)

MAE measures the mean absolute difference between predicted and true values and is suitable for regression problems.

### Mean Squared Error (MSE)

MSE calculates the mean squared difference between predicted and true values and is another loss function for regression tasks.

### Binary Cross-Entropy

Binary Cross-Entropy is commonly used for binary classification problems and measures the dissimilarity between predicted and true binary values.

### Categorical Cross-Entropy

Categorical Cross-Entropy is used for multi-class classification and quantifies the difference between predicted and true class probabilities.

## Convolution Methods

### Cross-Correlation with Padding

This repository includes an implementation of the cross-correlation method with various padding modes, including 'same', 'full', and 'valid'. You can choose the desired padding mode to control the output dimensions of convolutional layers.

### Pooling Methods

Pooling methods, such as Max Pooling, Average Pooling, and Global Average Pooling, are implemented to down-sample and reduce spatial dimensions in convolutional neural networks.

## Neural Network Implementation

### Network Class

The `Network` class provides a high-level interface for constructing neural networks. It allows you to stack layers, define the loss function, and train the model using backpropagation.

### Usage

1. Initialize a `Network` object.
2. Add layers using the `add` method.
3. Specify the loss function and its derivative using the `use` method.
4. Build the network using the `build` method (for shape initialization).
5. Train the model using the `fit` method.
6. Evaluate the model using the `predict` method.

## Example Usage

### XOR Data

An example demonstrates the network's ability to learn the XOR function.

### MNIST Data

The repository includes code to apply the network to the MNIST dataset for digit classification tasks.

Feel free to explore and experiment with the code to deepen your understanding of neural networks and deep learning.
