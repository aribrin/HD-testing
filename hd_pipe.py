import torch
from torch import nn
from torchvision import datasets, transforms

class HDTrainer():
    def __init__(self, input_size, output_size):
        self.weights = nn.Parameter(self.generate_weights(input_size, output_size))
        self.weights.requires_grad = True  # Enable gradient computation for the weights

    def generate_weights(self, input_size, output_size):
        # Generate random binary weights
        weights = torch.randint(0, 2, (input_size, output_size))
        weights = weights.float()
        weights[weights == 0] = -1
        return weights

    def forward(self, x):
        # Perform vector operations (e.g., circular convolution, permutation)
        # Here, we compute the dot product between the input and weights using batch matrix multiplication
        output = torch.matmul(x, self.weights)
        return output

    def train(self, train_loader):
        criterion = nn.CrossEntropyLoss()  # Use cross-entropy loss for multi-class classification
        optimizer = torch.optim.SGD([self.weights], lr=0.01)  # Use stochastic gradient descent optimizer

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            encoded_inputs = inputs.view(inputs.size(0), -1).sign()  # Perform binding by element-wise sign function
            output = self.forward(encoded_inputs)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            _, predicted = output.max(1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)

            accuracy = 100.0 * correct / total
            print('Accuracy: {:.2f}%'.format(accuracy))

def main(dataset_type):
    # Define the input and output dimensions based on the dataset type
    if dataset_type == "MNIST":
        input_size = 784  # MNIST images are 28x28 = 784
        output_size = 10  # MNIST has 10 classes (digits 0-9)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    elif dataset_type == "ISOLET":
        input_size = 617  # ISOLET has 617 input features
        output_size = 26  # ISOLET has 26 classes (letters A-Z)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = datasets.ISOLET('data', download=True, transform=transform)

    # Initialize the HDTrainer with the input and output dimensions
    trainer = HDTrainer(input_size, output_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Train the HD model
    trainer.train(train_loader)

if __name__ == '__main__':
    dataset_type = "ISOLET"  # Change this to "ISOLET" for the ISOLET dataset
    main(dataset_type)
