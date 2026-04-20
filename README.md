# Deep Learning Course Project: MNIST Classification
## Overview 
This project implements two neural network models from scratch for handwritten digit classification on the MNIST dataset:

- Task 1: Multi-Layer Perceptron 
- Task 2: Convolutional Neural Network

The goal is to understand the fundamentals of deep learning by manually implementing forward and backward propagation, experimenting with hyperparameters, and improving model performance under given constraints.

## Dataset
- MNIST Dataset
- 60,000 training images, 10,000 testing images
- Each image: 28 × 28 grayscale
- 10 classes (digits 0–9)

## Task 1: Multilayer Perceptron 
### Architecture
- Input layer: 784 neurons (flattened image)
- Hidden layer: 256 neurons (Sigmoid activation)
- Output layer: 10 neurons (Softmax)

### Inital Performance
- Accuracy: ~68%
- After Improvements: ~98%

### Improvements Made
- Increased learning rate: 0.1 -> 1.0
- Applied Xavier/He initialization
- Modified gradient updates:
  - Divided gradients individually instead of averaging improperly
### Key Learnings
- Proper weight initialization prevent vanishing/exploding gradients
- Learning rate signficantly impacts convergence
- Sigmoid activation can limit performance due to saturation

## Task 2: Convolutional Neural Network
### Constraint 
- Only 1 convolutional filter allowed

### Limitation
- Model learns only one feature map, severely limiting representation capacity

### Performance
- Accuracy: ~92%
- After Improvements: ~94%

### Improvements Made
- Added convolution bias
- Tuned learning rate:
  - Tested: 0.01, 0.5, 1.0
  - Best: 1.0
- Experimented with kernel sizes
  - Tested: 3, 5, 7, 9
  - Best: 7
- Adjusted weight scaling (2.0 -> 1.0)
- Did not use Xavier/He initialization
  
### Key Insight
- The primary bottleneck is the single filter constraint, not optimization
