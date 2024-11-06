import time
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# Define hyperparameters to try
learning_rates = [0.001, 0.0005, 0.0001]
batch_sizes = [4, 8, 16]

# Dictionary to store results
results = {
    "Learning Rate": [],
    "Batch Size": [],
    "Epoch": [],
    "Train Loss": [],
    "Test Loss": [],
    "Test Accuracy": []
}

# Loop through each combination of learning rate and batch size
for lr in learning_rates:
    for batch_size in batch_sizes:
        print(f"\nTraining with Learning Rate: {lr}, Batch Size: {batch_size}\n")

        # Load datasets and DataLoader with current batch size
        trainset = TeamMateDataset(n_images=50, train=True)
        testset = TeamMateDataset(n_images=10, train=False)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=1, shuffle=False)

        # Initialize model and optimizer
        model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        # Lists to store epoch-wise results
        train_losses = []
        test_losses = []
        test_accuracies = []

        # Epoch loop
        for epoch in range(1, 4):  # Reduced to 10 epochs for quicker testing

            # Start timer
            t = time.time_ns()

            # Train the model
            model.train()
            train_loss = 0

            # Batch Loop for Training
            for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):
                # Convert each tensor to PIL, apply transformation, and convert back to tensor
                images = [train_transforms(Image.fromarray(image.numpy().astype('uint8'))) for image in images]
                images = torch.stack(images).reshape(-1, 1, 64, 64).repeat(1, 3, 1, 1).to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)

                # Compute the loss
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Accumulate the loss
                train_loss += loss.item()

            # Test the model
            model.eval()
            test_loss = 0
            correct = 0
            total = 0

            # Batch Loop for Testing
            for images, labels in tqdm(testloader, total=len(testloader), leave=False):
                images = [transforms.ToTensor()(Image.fromarray(image.numpy().astype('uint8'))) for image in images]
                images = torch.stack(images).reshape(-1, 1, 64, 64).repeat(1, 3, 1, 1).to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()

                    # Get predicted class
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Calculate losses and accuracy
            avg_train_loss = train_loss / len(trainloader)
            avg_test_loss = test_loss / len(testloader)
            test_accuracy = correct / total

            # Print epoch results
            print(f"Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

            # Store the results for tabulation
            results["Learning Rate"].append(lr)
            results["Batch Size"].append(batch_size)
            results["Epoch"].append(epoch)
            results["Train Loss"].append(avg_train_loss)
            results["Test Loss"].append(avg_test_loss)
            results["Test Accuracy"].append(test_accuracy)

# Plotting the results for each learning rate and batch size combination
plt.figure(figsize=(10, 5))
for lr in learning_rates:
    for batch_size in batch_sizes:
        epochs = [e for i, e in enumerate(results["Epoch"]) if results["Learning Rate"][i] == lr and results["Batch Size"][i] == batch_size]
        train_loss = [t for i, t in enumerate(results["Train Loss"]) if results["Learning Rate"][i] == lr and results["Batch Size"][i] == batch_size]
        test_loss = [t for i, t in enumerate(results["Test Loss"]) if results["Learning Rate"][i] == lr and results["Batch Size"][i] == batch_size]
        
        plt.plot(epochs, train_loss, label=f'LR: {lr}, Batch: {batch_size} Train Loss')
        plt.plot(epochs, test_loss, label=f'LR: {lr}, Batch: {batch_size} Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Testing Losses for Different Learning Rates and Batch Sizes')
plt.savefig('lab8/task6_loss_comparison_plot.png')

# Show accuracy comparison
plt.figure(figsize=(10, 5))
for lr in learning_rates:
    for batch_size in batch_sizes:
        epochs = [e for i, e in enumerate(results["Epoch"]) if results["Learning Rate"][i] == lr and results["Batch Size"][i] == batch_size]
        test_accuracy = [a for i, a in enumerate(results["Test Accuracy"]) if results["Learning Rate"][i] == lr and results["Batch Size"][i] == batch_size]
        plt.plot(epochs, test_accuracy, label=f'LR: {lr}, Batch: {batch_size} Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.legend()
plt.title('Test Accuracy for Different Learning Rates and Batch Sizes')
plt.savefig('lab8/task6_accuracy_comparison_plot.png')

print("Training completed for all learning rates and batch sizes. Plots saved.")
