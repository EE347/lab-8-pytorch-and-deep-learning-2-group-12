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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations with data augmentations
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# Optimal hyperparameters found in Task 6
learning_rate = 0.0005  # Example optimal learning rate
batch_size = 8          # Example optimal batch size

# Load datasets and DataLoader
trainset = TeamMateDataset(n_images=50, train=True)
testset = TeamMateDataset(n_images=10, train=False)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

# Initialize model, optimizer, and loss function
model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Lists to store results
train_losses = []
test_losses = []
test_accuracies = []

# Epoch loop
num_epochs = 4 # Set to a high number if needed for improved performance
for epoch in range(1, num_epochs + 1):

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
    all_labels = []
    all_predictions = []

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

            # Collect all labels and predictions for the confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate losses and accuracy
    avg_train_loss = train_loss / len(trainloader)
    avg_test_loss = test_loss / len(testloader)
    test_accuracy = correct / total

    # Print epoch results
    print(f"Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Store the results for tabulation
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    test_accuracies.append(test_accuracy)

    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix for Epoch {epoch}")
    plt.savefig(f'lab8/confusion_matrix_epoch_{epoch}.png')
    plt.close()

# Save the final model
torch.save(model.state_dict(), 'lab8/final_model.pth')
print("Training completed. Final model saved as 'lab8/final_model.pth'.")
