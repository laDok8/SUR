import numpy as np
import ikrlib as ilib

CLASSES = 31

class GMM():
    def __init__(self) -> None:
        self.mean_face = self.calculate_mean_face()


    def calculate_mean_face(self):
        train = {}
        dev = {}
        for i in range(1, CLASSES + 1):
            train[i] = np.array(list(ilib.png_load(f"train2/{i}/da", True).values())).reshape(-1, 80 * 80)
            dev[i] = np.array(list(ilib.png_load(f"dev2/{i}", True).values())).reshape(-1, 80 * 80)
            print("Loading data was successful")

        # Concatenate all class data
        train_all = np.concatenate([train[i] for i in range(1, CLASSES + 1)], axis=0)

        # Calculate mean face
        mean_face = np.mean(train_all, axis=0)
        return mean_face
    
    def train_gmm(self):
        #for i in range(1,CLASSES+1):
        #    augment_images(f"train/{i}", f"train/{i}/da", 3)
        #    augment_images(f"dev/{i}", f"dev/{i}/da", 3)

        train = {}
        dev = {}
        for i in range(1,CLASSES+1):
            train[i] = np.array(list(ilib.png_load(f"train2/{i}/da", True).values())).reshape(-1, 80 * 80)
            dev[i] = np.array(list(ilib.png_load(f"dev2/{i}", True).values())).reshape(-1, 80 * 80)
            print("Loading data was successful")

        # Concatenate all class data
        train_all = np.concatenate([train[i] for i in range(1, CLASSES + 1)], axis=0)

        # Calculate mean face
        mean_face = np.mean(train_all, axis=0)

        dev_subs_mean = {}
        train_subs_mean = {}
        for i in range(1, CLASSES+1):
            print(f"Creating subs mean class {i}")
            data_subs_mean = train[i] - mean_face
            V, S, U = np.linalg.svd(train_all, full_matrices=False)
            train_subs_mean[i] = data_subs_mean.dot(U.T)

            dev_subs_mean[i] = (dev[i] - mean_face).dot(U.T)

        # Train GMM for each class
        num_components = 2
        ws_list = []
        mus_list = []
        covs_list = []

        for i in range(1, CLASSES + 1):
            print(f"Training GMM class {i}")
            data = train_subs_mean[i]
            init_ws = np.ones(num_components) / num_components
            init_mus = data[np.random.choice(len(data), num_components, replace=False)]
            init_covs = np.array([np.eye(data.shape[1]) * 1e-2 for _ in range(num_components)])
            ws, mus, covs, _ = ilib.train_gmm(data, init_ws, init_mus, init_covs)

            ws_list.append(ws)
            mus_list.append(mus)
            covs_list.append(covs)

        return dev_subs_mean, ws_list, mus_list, covs_list
    
    def classify_gmm(self, x, ws_list, mus_list, covs_list):
        log_probs = np.array([ilib.logpdf_gmm(x, ws, mus, covs) for ws, mus, covs in zip(ws_list, mus_list, covs_list)])
        return np.argmax(log_probs, axis=0)

    def eval(self, dev_subs_mean, ws_list, mus_list, covs_list):
        # Classify test images
        dev_true_labels = []
        dev_predicted_labels = []

        for i in range(1, CLASSES + 1):
            print(f"Evaluating GMM class {i}")
            dev_true_labels.extend([i - 1] * len(dev_subs_mean[i]))
            dev_predicted_labels.extend(self.classify_gmm(dev_subs_mean[i], ws_list, mus_list, covs_list))

        dev_true_labels = np.array(dev_true_labels)
        dev_predicted_labels = np.array(dev_predicted_labels)

        # Calculate accuracy
        accuracy = np.sum(dev_true_labels == dev_predicted_labels) / len(dev_true_labels)
        print("Accuracy:", accuracy)