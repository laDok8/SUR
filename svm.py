import numpy as np
from sklearn import svm
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


class SVCTrain():

    def toBlackWhite(self, image_arr):
        return np.dot(image_arr[..., :3], [0.299, 0.587, 0.114])

    def __init__(self) -> None:
        self.clf = svm.SVC(kernel='linear', probability=True)

    def train_svc(self, train_dataset, test_dataset):
        transposed = np.transpose(train_dataset.images, (0, 2, 3, 1))
        train_x = np.array(list(map(self.toBlackWhite, transposed))).reshape(-1, 80 * 80)
        self.clf.fit(train_x, train_dataset.labels)
        predicted = self.clf.predict(train_x)
        correct = np.sum(predicted == train_dataset.labels)
        accuracy = correct / len(test_dataset.labels)

    def predict(self, data):
        transposed = np.transpose(data, (1, 2, 0))
        new_d = np.array(list(map(self.toBlackWhite, transposed))).reshape(-1, 80 * 80)
        return self.clf.predict_proba(new_d)
