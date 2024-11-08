import time
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from PIL import Image  # Import PIL to handle image format conversion
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small

if __name__ == '__main__':

    # Set device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the transformation
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Apply random horizontal flip
        transforms.ToTensor(),
    ])

    # Load datasets without transformations in the initialization
    trainset = TeamMateDataset(n_images=50, train=True)
    testset = TeamMateDataset(n_images=10, train=False)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # Create the model and optimizer
    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    # Saving parameters
    best_train_loss = 1e9

    # Loss lists
    train_losses = []
    test_losses = []

    # Epoch Loop
    for epoch in range(1, 100):

        # Start timer
        t = time.time_ns()

        # Train the model
        model.train()
        train_loss = 0

        # Batch Loop for Training
        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):

            # Convert each tensor to PIL, apply transformation, and convert back to tensor
            images = [train_transforms(Image.fromarray(image.numpy().astype('uint8'))) for image in images]

            # Stack images and reshape to include 3 channels by repeating the grayscale channel
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

            # Update the parameters
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

            # Convert each tensor to PIL, apply transformation, and convert back to tensor
            images = [transforms.ToTensor()(Image.fromarray(image.numpy().astype('uint8'))) for image in images]

            # Stack images and reshape to include 3 channels by repeating the grayscale channel
            images = torch.stack(images).reshape(-1, 1, 64, 64).repeat(1, 3, 1, 1).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Accumulate the loss
            test_loss += loss.item()

            # Get the predicted class from the maximum value in the output-list of class scores
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)

            # Accumulate the number of correct classifications
            correct += (predicted == labels).sum().item()

        # Print the epoch statistics
        print(f'Epoch: {epoch}, Train Loss: {train_loss / len(trainloader):.4f}, Test Loss: {test_loss / len(testloader):.4f}, Test Accuracy: {correct / total:.4f}, Time: {(time.time_ns() - t) / 1e9:.2f}s')

        # Update loss lists
        train_losses.append(train_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        # Update the best model
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), 'lab8/best_model.pth')

        # Save the model
        torch.save(model.state_dict(), 'lab8/current_model.pth')

        # Create the loss plot
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('lab8/task3_loss_plot.png')
