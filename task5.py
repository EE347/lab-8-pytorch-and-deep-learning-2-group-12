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

# Define transformations with RandomHorizontalFlip and RandomRotation
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# Load datasets without transformations in the initialization
trainset = TeamMateDataset(n_images=50, train=True)
testset = TeamMateDataset(n_images=10, train=False)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

# Define two loss functions
loss_functions = {
    "CrossEntropyLoss": torch.nn.CrossEntropyLoss(),
    "NLLLoss": torch.nn.NLLLoss()
}

# Dictionary to store results
results = {
    "Loss Function": [],
    "Epoch": [],
    "Train Loss": [],
    "Test Loss": [],
    "Test Accuracy": []
}

# Loop through each loss function
for loss_name, criterion in loss_functions.items():
    print(f"\nTraining with {loss_name}...\n")

    # Initialize model and optimizer
    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

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
        results["Loss Function"].append(loss_name)
        results["Epoch"].append(epoch)
        results["Train Loss"].append(avg_train_loss)
        results["Test Loss"].append(avg_test_loss)
        results["Test Accuracy"].append(test_accuracy)

# Plotting the results
plt.figure(figsize=(10, 5))
for loss_name in loss_functions:
    epochs = [e for i, e in enumerate(results["Epoch"]) if results["Loss Function"][i] == loss_name]
    train_loss = [t for i, t in enumerate(results["Train Loss"]) if results["Loss Function"][i] == loss_name]
    test_loss = [t for i, t in enumerate(results["Test Loss"]) if results["Loss Function"][i] == loss_name]
    test_accuracy = [a for i, a in enumerate(results["Test Accuracy"]) if results["Loss Function"][i] == loss_name]
    
    plt.plot(epochs, train_loss, label=f'{loss_name} Train Loss')
    plt.plot(epochs, test_loss, label=f'{loss_name} Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Testing Losses for Different Loss Functions')
plt.savefig('lab8/task5_loss_comparison_plot.png')

# Show accuracy comparison
plt.figure(figsize=(10, 5))
for loss_name in loss_functions:
    epochs = [e for i, e in enumerate(results["Epoch"]) if results["Loss Function"][i] == loss_name]
    test_accuracy = [a for i, a in enumerate(results["Test Accuracy"]) if results["Loss Function"][i] == loss_name]
    plt.plot(epochs, test_accuracy, label=f'{loss_name} Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.legend()
plt.title('Test Accuracy for Different Loss Functions')
plt.savefig('lab8/task5_accuracy_comparison_plot.png')

print("Training completed for both loss functions. Plots saved.")
