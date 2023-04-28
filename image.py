import os
import torch
import Augmentor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu" 

data_augmentation_enabled = True
CLASSES = 31

def augment_images(input_dir, output_dir, num_augmentations=int(1e3)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for cls in range(1, CLASSES+1):
        in_dir = input_dir + '/' + str(cls)
        print('Augmenting images in ' + in_dir)
        out_dir = output_dir + '/' + str(cls)
        print('Augmenting into ' + out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        p = Augmentor.Pipeline(source_directory=in_dir, output_directory="out")
        p.random_brightness(probability=0.4, min_factor=0.01, max_factor=0.5)
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        p.zoom_random(probability=0.5, percentage_area=0.8)
        p.flip_left_right(probability=0.3)
        p.skew_tilt(probability=0.5, magnitude=0.1)
        p.sample(num_augmentations)
        #out is relative for augmentator :/ move
        os.rename(os.path.join(input_dir,str(cls),'out'), os.path.join(output_dir,str(cls)))

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

class SmallCNNMultiClass(nn.Module):
    def __init__(self, num_classes=CLASSES):
        super(SmallCNNMultiClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm1 = nn.BatchNorm2d(8)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout2d(0.4)
        self.fc1 = nn.Linear(32 * 10 * 10, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        #VGG like
        x = self.pool(F.relu(self.conv1(x)))  # 40x40x8
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))  # 20x20x16
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))  # 10x10x32
        x = x.view(-1, 32 * 10 * 10)  # 3200
        x = self.fc1(x)  # 128
        x = self.fc2(x)
        return x

def fit(num_epochs, model, optimizer, criterion, train_loader, dev_loader):
    # Training loop
    losses = []
    accuracys = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if not epoch % 10:
        # Evaluation on the dev set
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in dev_loader:
                    inputs, labels = inputs.to(dev), labels.to(dev)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = correct / total
            losses.append(train_loss)
            accuracys.append(accuracy)
            print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}, {correct} and {total}')

    plt.figure()
    plt.plot(accuracys)

    plt.figure()
    plt.plot(losses)

def eval(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(dev), labels.to(dev)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}, {correct} and {total}')

def predict_image(image, model, dev):
    xb = image.unsqueeze(0).to(dev)
    yb = model(xb)
    yb = yb.to(dev)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()