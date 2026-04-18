import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_MNIST import load_data

# ===================== Utility Functions ===================== #

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    

# ===================== Data Loading ===================== #
def dataloader(train_dataset, test_dataset, batch_size=64):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)
    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))
    return dataloader(train_dataset, test_dataset)

# ===================== CNN Structure ===================== #
class CNN:
    def __init__(self, input_size, num_filters, kernel_size, fc_output_size, lr):
        self.lr = lr
        self.jernel_size = kernel_size
        self.num_filters = num_filters

        self.conv_output_size = input_size - kernel_size + 1
        self.flat_size = self.conv_output_size * self.conv_output_size * num_filters
        self.kernel = np.random.randn(kernel_size, kernel_size, num_filters) * np.sqrt(2.0 / (kernel_size * kernel_size))

        self.fc_w = np.random.randn(self.flat_size, fc_output_size) * np.sqrt(2.0 / self.flat_size)
        self.fc_b = np.zeros((1, fc_output_size))
        

    def forward(self, x):
        N = x.shape[0]
        k = self.kernel_size
        c = self.conv_output_size

        self.conv_output = np.zeros(N, c, c, self.num_filters)

        for i in range(c):
            for j in range(c):
                patch = x[:, i:i+k, j:j+k]
                for f in range(self.num_filters):
                    self.conv_output[:, i, j, f] = np.sum(patch * self.kernel[:, :, f], axis=(1, 2))
        
        self.relu_output = relu(self.conv_output)
        self.flat_output = self.relu_output.reshape(N, -1)
        self.fc_output = self.flat_output @ self.fc_w + self.fc_b
        outputs = softmax(self.fc_output)

        return outputs

    def backward(self, x, y, pred):
        """ Backward propagation """
        # 1. one-hot encode the labels

        # 2. Calculate softmax cross-entropy loss gradient
        
        # 3. Calculate fully connected layer gradient
        
        # 4. Backpropagate through ReLU
        
        # 5. Calculate convolution kernel gradient
        
        # 6. Update parameters
        

    def train(self, x, y):
        # call forward function
        
        # calculate loss
        
        # call backward function

        return loss

# ===================== Training Process ===================== #
def main():
    # First, load data
    train_loader, test_loader = load_data()

    # Second, define hyperparameters
    input_size = 28
    num_filters = 1
    kernel_size = 5
    fc_output_size = 10
    lr = 0.01
    num_epochs = 5

    model = CNN(input_size, num_filters, kernel_size, fc_output_size, lr)
    

    # Then, train the model
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, lables in train_loader:  # define training phase for training model
            x = inputs.numpy().squeeze(1)
            y = lables.numpy()
            total_loss += model.train(x, y)
            

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}") # print the loss for each epoch

    # Finally, evaluate the model
    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.numpy().squeeze(1)
        y = labels.numpy()
        pred = model.forward(x)  # the model refers to the model that was trained during the raining phase
        predicted_labels = np.argmax(pred, 1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    print(f"Test Accuracy: {correct_pred/total_pred}")

if __name__ == "__main__":
    main()