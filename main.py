import torch
from torch import nn
from torchvision import datasets, transforms

class Trainer():
    def __init__(self):
        # Define hyperparameters and other necessary variables
        self.input_dim = 784  # Input dimension for MNIST dataset
        self.hidden_dim = 100  # Dimension of the hyperdimensional space
        self.num_classes = 10  # Number of output classes
        self.learning_rate = 0.001
        self.num_epochs = 10
        self.batch_size = 64

        # Define the model architecture
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

        # Define the loss function
        self.criterion = nn.CrossEntropyLoss()

        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def forward(self, x):
        # Forward pass through the model
        output = self.model(x)
        return output

    def train(self, train_loader):
        # Set the model to training mode
        self.model.train()

        # Iterate over the training dataset for the specified number of epochs
        for epoch in range(self.num_epochs):
            # Iterate over batches of data
            for batch_idx, (data, target) in enumerate(train_loader):
                # Move the data to the device (e.g., CPU or GPU)
                data, target = data.to(device), target.to(device)

                # Flatten the input data
                data = data.view(data.size(0), -1)

                # Forward pass
                output = self.forward(data)

                # Calculate the loss
                loss = self.criterion(output, target)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Print training progress
                if batch_idx % 100 == 0:
                    print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

    def test(self, test_loader):
        # Set the model to evaluation mode
        self.model.eval()

        # Initialize variables for accuracy calculation
        test_loss = 0
        correct = 0

        # Disable gradient computation
        with torch.no_grad():
            # Iterate over the test dataset
            for data, target in test_loader:
                # Move the data to the device
                data, target = data.to(device), target.to(device)

                # Flatten the input data
                data = data.view(data.size(0), -1)

                # Forward pass
                output = self.forward(data)

                # Calculate the loss
                test_loss += self.criterion(output, target).item()

                # Get the predicted classes
                _, predicted = output.max(1)

                # Count the number of correct predictions
                correct += predicted.eq(target).sum().item()

        # Calculate the average loss and accuracy
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        # Print the test results
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), accuracy))

def main():
    # Set the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=trainer.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=trainer.batch_size, shuffle=False)

    # Create an instance of the Trainer class
    trainer = Trainer()

    # Move the model and data to the device
    trainer.model.to(device)
    train_loader.dataset.data = train_loader.dataset.data.to(device)
    test_loader.dataset.data = test_loader.dataset.data.to(device)

    # Train the model
    trainer.train(train_loader)

    # Test the model
    trainer.test(test_loader)

if __name__ == '__main__':
    main()

