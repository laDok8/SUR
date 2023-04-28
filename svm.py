from torch.utils.data import Dataset
import os
import numpy as np
from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
from sklearn import svm

CLASSES = 31


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

class SVMTrain():

    def toBlackWhite(self, image_arr):
        return np.dot(image_arr[...,:3], [0.299, 0.587, 0.114])

    def __init__(self) -> None:
        self.clf = svm.SVC(kernel='linear')

    def train(self, train_dataset, test_dataset):
        # train_dataset (80,80,3) -> (0,6400) using BlackWhite and reshape
        train_x = np.array(list(map(self.toBlackWhite, train_dataset.images))).reshape(-1, 80 * 80)
        self.clf.fit(train_x, train_dataset.labels)
        predicted = self.clf.predict(test_dataset.images.reshape(-1, 80 * 80))
        correct = (predicted == test_dataset.labels).sum().item()
        accuracy = correct / len(test_dataset.labels)
        print(f'Accuracy: {accuracy:.4f}, {correct} and {len(test_dataset.labels)}')

    def predict(self, data):
        data = np.array(list(map(self.toBlackWhite, data))).reshape(-1, 80 * 80)
        return self.clf.predict(data)