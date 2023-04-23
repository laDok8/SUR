import os
import torch
import random

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from glob import glob
from ikrlib import png2fea
from augment import augment_images
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, images, labels) -> None:
        self.images = images
        self.labels = labels

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]      
        return image, label

    def __len__(self):
        return len(self.labels)

CLASSES = 31

# augment_images('dev', 'dev/da')
# augment_images('train', 'train/da')

def png2fea2(dir_name):
    """
    Loads all *.png images from directory dir_name into a dictionary. Keys are the file names
    and values and 2D numpy arrays with corresponding grayscale images
    """
    features = {}
    for f in glob(dir_name + '/*.png'):
        features[f] = np.array(Image.open(f).convert('L'), dtype=np.float64)
    return features

train_x = np.empty((0,80,80))
train_y = np.empty((0),dtype=int)

test_x = np.empty((0,80,80))
test_y = np.empty((0),dtype=int)

for i in range(1,CLASSES+1): 
    train_i = np.array(list(png2fea2('dev/da/'+str(i)).values()))
    label_i = np.full(len(train_i),i-1)
    train_x = np.concatenate((train_x, train_i), axis=0)
    train_y = np.concatenate((train_y,label_i), axis=0)

    test_i = np.array(list(png2fea2('test/da'+str(i)).values()))
    label_i = np.full(len(test_i),i-1)
    test_x = np.concatenate((test_x, train_i), axis=0)
    test_y = np.concatenate((train_y,label_i), axis=0)

print("Images were successfully loaded")

train_x = np.array(train_x)
test_x = np.array(test_x)

class SmallCNNMultiClass(nn.Module):
    def __init__(self, num_classes=31):
        super(SmallCNNMultiClass, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(32 * 20 * 20, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 20 * 20)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Convert NumPy arrays to PyTorch tensors
train_tensors = torch.Tensor(train_x).unsqueeze(1)
dev_tensors = torch.Tensor(test_x).unsqueeze(1)

# Create new TensorDataset instances with the modified labels
train_dataset = CustomDataset(train_tensors, train_y)
dev_dataset = CustomDataset(dev_tensors, train_y)

# import random
# x, y = random.choice(train_dataset)
# print(y)

# import matplotlib.pyplot as plt

# plt.imshow(x[0,:,:])
# plt.savefig('idk.png')

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

model = SmallCNNMultiClass()
criterion = F.cross_entropy
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
    train_loss = train_loss / len(train_loader.dataset)

    # Evaluation on the dev set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dev_loader:
            outputs = model(inputs)
            predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += torch.sum(predicted.values == labels).item() / len(predicted)

    accuracy = correct / total
    print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}')