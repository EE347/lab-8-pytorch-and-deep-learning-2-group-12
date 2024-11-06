# In utils/dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset

class TeamMateDataset(Dataset):
    def __init__(self, n_images, train=True):
        dataset_type = "train" if train else "test"
        self.images = []
        self.labels = []

        for label in [0, 1]:  # Assuming two classes, 0 and 1
            folder_path = f'/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-12/data/{dataset_type}/{label}'
            for i, filename in enumerate(os.listdir(folder_path)):
                if i >= n_images:
                    break
                img_path = os.path.join(folder_path, filename)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    self.images.append(image)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label)
