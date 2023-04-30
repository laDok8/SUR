import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


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


class CNNModel(nn.Module):
    class SmallCNNMultiClass(nn.Module):
        def __init__(self, num_classes=31):
            super(CNNModel.SmallCNNMultiClass, self).__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.batch_norm1 = nn.BatchNorm2d(8)
            self.batch_norm2 = nn.BatchNorm2d(16)
            self.dropout = nn.Dropout2d(0.4)
            self.fc1 = nn.Linear(32 * 10 * 10, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.soft_max = nn.Softmax(dim=0)

        def forward(self, x):
            # VGG like
            x = self.pool(F.relu(self.conv1(x)))  # 40x40x8
            x = self.dropout(x)
            x = self.pool(F.relu(self.conv2(x)))  # 20x20x16
            x = self.dropout(x)
            x = self.pool(F.relu(self.conv3(x)))  # 10x10x32
            x = x.view(-1, 32 * 10 * 10)  # 3200
            x = self.fc1(x)  # 128
            x = self.fc2(x)
            return x

    def __init__(self, num_classes=31, lr=int(1e-5)):
        super(CNNModel, self).__init__()
        self.model = self.SmallCNNMultiClass(num_classes)
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        self.dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    def train_net(self, train_dataset, test_dataset, num_epochs=200):
        batch_size = 16
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        losses = []
        accuracys = []
        for epochs in range(num_epochs + 1):
            self.model.train()
            train_loss = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.dev), labels.to(self.dev)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            if not epochs % 50:
                # Evaluation on the dev set
                self.model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in dev_loader:
                        inputs, labels = inputs.to(self.dev), labels.to(self.dev)
                        outputs = self.model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                accuracy = correct / total
                losses.append(train_loss)
                accuracys.append(accuracy)
                print(f'Epoch: {epochs}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}, {correct} and {total}')

        return accuracys, losses

    def predict(self, dataset):
        # return probabilities for all classes
        # to get max ->  _, predicted = torch.max(outputs.data, 1)
        self.model.eval()
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        return [self.model(inputs.to(self.dev)) for inputs, _ in loader]
