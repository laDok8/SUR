import torch
import numpy as np
import torch.nn as nn
from numpy.random import randint
from torch.utils.data import Dataset

import ikrlib as ilib


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


class GMMmodel(nn.Module):
    def __init__(self, num_classes=31):
        super(GMMmodel, self).__init__()
        self.num_classes = num_classes
        self.ws = []
        self.mus = []
        self.covs = []

    def toBlackWhite(self, image_arr):
        # (186, 3, 80, 80) -> (186,6400) using BlackWhite and reshape
        return np.dot(image_arr[..., :3], [0.299, 0.587, 0.114])

    def predict(self, data):
        if isinstance(data, torch.Tensor):
            data = np.array(data).transpose(0,2,3,1)
            data = self.toBlackWhite(data)
            data = data.reshape(-1,80*80)

        log_probs = np.array(
            [ilib.logpdf_gmm(data, self.ws[i], self.mus[i], self.covs[i]) for i in range(self.num_classes)])
        # return np.argmax(log_probs, axis=0) + 1

        return log_probs

    def train_gmm(self, train_dataset, test_dataset, eval_dataset):
        train_dataset_np = np.array(train_dataset.images).transpose(0, 2, 3, 1)
        test_dataset_np = np.array(test_dataset.images).transpose(0, 2, 3, 1)
        eval_dataset_np = np.array(eval_dataset.images).transpose(0, 2, 3, 1)

        # to black and white
        train_dataset_np = self.toBlackWhite(train_dataset_np)
        test_dataset_np = self.toBlackWhite(test_dataset_np)
        eval_dataset_np = self.toBlackWhite(eval_dataset_np)

        # (186, 80, 80) -> (186, 6400)
        train_dataset_np = train_dataset_np.reshape(-1, 80 * 80)
        test_dataset_np = test_dataset_np.reshape(-1, 80 * 80)
        eval_dataset_np = eval_dataset_np.reshape(-1, 80 * 80)

        # separate classes
        train = [[train_dataset_np[i] for i in range(len(train_dataset_np)) if train_dataset.labels[i] == j] for j in
                 range(0, self.num_classes)]
        dev = [[test_dataset_np[i] for i in range(len(test_dataset_np)) if test_dataset.labels[i] == j] for j in
               range(0, self.num_classes)]
        eval_data = [[eval_dataset_np[i] for i in range(len(eval_dataset_np)) if eval_dataset.labels[i] == j] for j in
               range(0, self.num_classes)]

        # Concatenate all class data
        train_all = np.concatenate([train[i] for i in range(0, self.num_classes)], axis=0)
        # Calculate mean face
        mean_face = np.mean(train_all, axis=0)

        dev_subs_mean = {}
        train_subs_mean = {}
        eval_subs_mean = {}
        print(f"Creating subs mean classes SVD")
        V, S, U = np.linalg.svd(train_all, full_matrices=False)
        for i in range(0, self.num_classes):
            train_subs_mean[i] = (train[i] - mean_face).dot(U.T)
            dev_subs_mean[i] = (dev[i] - mean_face).dot(U.T)

        eval_subs_mean[0] = (eval_data[0] - mean_face).dot(U.T)

        print(f"Training GMM")
        for i in range(0, self.num_classes):
            data = train_subs_mean[i]
            init_ws = np.ones(self.num_classes) / self.num_classes
            init_mus = data[randint(0, len(data), self.num_classes)]
            init_covs = [np.eye(data.shape[1])] * self.num_classes
            ws_m, mus_m, covs_m, _ = ilib.train_gmm(data, init_ws, init_mus, init_covs)

            self.ws.append(ws_m)
            self.mus.append(mus_m)
            self.covs.append(covs_m)

        for jj in range(10):
            # TTL_t je doveryhodnosť
            for i in range(0, self.num_classes):
                data = train_subs_mean[i]

                self.ws[i], self.mus[i], self.covs[i], TTL = ilib.train_gmm(data, self.ws[i], self.mus[i], self.covs[i])
                print(f'Iteration: {jj} Total log likelihood: {TTL} for person {i}')

        corect = 0
        for i in range(0, self.num_classes):
            labels_i = np.argmax(self.predict(dev_subs_mean[i]), axis=0)
            corect += np.sum(labels_i == i)
        print(f"Accuracy: {corect / len(test_dataset_np)}")

        return eval_subs_mean
