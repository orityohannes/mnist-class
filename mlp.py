from read_MNIST import load_data

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def dataloader(train_dataset, test_dataset, batch_size=128):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)
    print("The number of training data:", len(train_dataset))
    print("The number of testing data:", len(test_dataset))
    return dataloader(train_dataset, test_dataset)

class MLP:
    def __init__(self, input_size, hidden_size, output_size,lr):
        self.lr = lr

        # LAYER 1
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        # LAYER 2
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    
    def forward(self, x):  # forward propagation to get predictions
        
        # LAYER 1
        self.z1 = x @ self.w1 + self.b1
        self.a1 = sigmoid(self.z1)

        # LAYER 2
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = softmax(self.z2)
        
        return self.a2
    
    def backward(self, x, y, pred):
        N = x.shape[0]

        # one-hot encode the labels
        y_onehot = np.zeros_like(pred)
        y_onehot[np.arange(N), y] = 1
        delta2 = (pred - y_onehot) / N

        # compute the gradients
        dw2 = (self.a1.T @ delta2) / N
        db2 = np.sum(delta2, axis=0, keepdims=True) / N
        sigmoid_deriv = self.a1 * (1 - self.a1)
        delta1 = (delta2 @ self.w2.T) * sigmoid_deriv
        dw1 = (x.T @ delta1) / N
        db1 = np.sum(delta1, axis=0, keepdims=True) / N
        
        # update the weights and biases
        self.w2 = self.w2 - (self.lr * dw2)
        self.b2 = self.b2 - (self.lr * db2)
        self.w1 = self.w1 - (self.lr * dw1)
        self.b1 = self.b1 - (self.lr * db1)
        

    def train(self, x,y):
        # call forward function
        
        # calculate loss
        
        # call backward function

        return loss

def main():
    # First, load data
    train_loader, test_loader = load_data()

    # Second, define hyperparameters
    input_size = 28*28  # MNIST images are 28x28 pixels
    hidden_size = 128
    output_size = 10
    lr = 0.1
    num_epochs = 100


    # Then, train the model
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, lables in train_loader:  # define training phase for training model
            

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}") # print the loss for each epoch

    # Finally, evaluate the model
    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.view(-1, input_size).numpy()
        y = labels.numpy()
        pred = model.forward(x)  # the model refers to the model that was trained during the raining phase
        predicted_labels = np.argmax(pred, 1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    print(f"Test Accuracy: {correct_pred/total_pred}")

if __name__ == "__main__":  # Program entry
    main()  