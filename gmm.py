import os
import numpy as np

from glob import glob
from PIL import Image
from PIL import ImageOps
from numpy.random import randint
from matplotlib import pyplot as plt
from numpy.linalg import inv
import ikrlib as ilib
import Augmentor
from PIL import ImageEnhance

from augment import augment_images
import random


CLASSES = 31
data_augmentation_enabled = True

def png_load(dir_name):
    """
    Loads all *.png images from directory dir_name into a dictionary. Keys are the file names
    and values and 2D numpy arrays with corresponding grayscale images
    """
    features = {}
    for f in glob(dir_name + '/*.png'):
        features[f] = np.array(Image.open(f).convert('L'), dtype=np.float64)
    return features

def random_rotation(image):
    angle = random.randint(-30, 30)
    return image.rotate(angle)

def random_flip(image):
    flip_type = random.choice(["horizontal", "vertical"])
    if flip_type == "horizontal":
        return image.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
    else:
        return image.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)

def random_contrast(image):
    factor = random.uniform(0.4, 1.7)
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def augment_image(image):
    image = random_rotation(image)
    image = random_flip(image)
    image = random_contrast(image)
    return image

def random_scale(image):
    scale_factor = random.uniform(0.9, 1.1)
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    image = image.resize((new_width, new_height), Image.LANCZOS)

    # Padding to maintain original size
    pad_x = max(0, width - new_width)
    pad_y = max(0, height - new_height)
    image = ImageOps.expand(image, (pad_x // 2, pad_y // 2, pad_x - pad_x // 2, pad_y - pad_y // 2))
    return image

def augment_images_gmm(input_dir, output_dir, num_augmentations=3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".png"):
            image_path = os.path.join(input_dir, file_name)
            image = Image.open(image_path)

            for i in range(num_augmentations):
                augmented_image = augment_image(image)
                output_path = os.path.join(output_dir, f"{file_name[:-4]}_aug_{i}.png")
                augmented_image.save(output_path)
                print(f"Data augumentation, new image created: {output_path}")

def get_train_all():
    train = {}
    dev = {}
    for i in range(1, CLASSES + 1):
        train[i] = np.array(list(png_load(f"train/{i}").values())).reshape(-1, 80 * 80)
        dev[i] = np.array(list(png_load(f"dev/{i}").values())).reshape(-1, 80 * 80)
    print("Loading data was successful")

    # Concatenate all class data
    train_all = np.concatenate([train[i] for i in range(1, CLASSES + 1)], axis=0)
    return train_all

def calculate_mean_face():
    train = {}
    dev = {}
    for i in range(1, CLASSES + 1):
        train[i] = np.array(list(png_load(f"train/{i}").values())).reshape(-1, 80 * 80)
        dev[i] = np.array(list(png_load(f"dev/{i}").values())).reshape(-1, 80 * 80)
    print("Loading data was successful")

    # Concatenate all class data
    train_all = np.concatenate([train[i] for i in range(1, CLASSES + 1)], axis=0)

    # Calculate mean face
    mean_face = np.mean(train_all, axis=0)
    return mean_face

def train_gmm():
    #for i in range(1,CLASSES+1):
    #    augment_images(f"train/{i}", f"train/{i}", 3)
    #    augment_images(f"dev/{i}", f"dev/{i}", 3)

    train = {}
    dev = {}
    for i in range(1,31+1):
        train[i] = np.array(list(png_load(f"train/{i}").values())).reshape(-1, 80 * 80)
        dev[i] = np.array(list(png_load(f"dev/{i}").values())).reshape(-1, 80 * 80)
    print("Loading data was successful")

    # Concatenate all class data
    train_all = np.concatenate([train[i] for i in range(1, CLASSES + 1)], axis=0)

    # Calculate mean face
    mean_face = np.mean(train_all, axis=0)

    dev_subs_mean = {}
    train_subs_mean = {}
    print(f"Creating subs mean classes")
    for i in range(1, CLASSES+1):
        data_subs_mean = train[i] - mean_face
        V, S, U = np.linalg.svd(train_all, full_matrices=False)
        train_subs_mean[i] = data_subs_mean.dot(U.T)

        dev_subs_mean[i] = (dev[i] - mean_face).dot(U.T)

    # Train GMM for each class
    num_components = 2
    ws_list = []
    mus_list = []
    covs_list = []

    print(f"Training GMM")
    for i in range(1, CLASSES + 1):
        data = train_subs_mean[i]
        init_ws = np.ones(num_components) / num_components
        init_mus = data[np.random.choice(len(data), num_components, replace=False)]
        init_covs = np.array([np.eye(data.shape[1]) * 1e-2 for _ in range(num_components)])
        ws, mus, covs, _ = ilib.train_gmm(data, init_ws, init_mus, init_covs)

        ws_list.append(ws)
        mus_list.append(mus)
        covs_list.append(covs)

    return dev_subs_mean, ws_list, mus_list, covs_list

def classify_gmm(x, ws_list, mus_list, covs_list):
    log_probs = np.array([ilib.logpdf_gmm(x, ws, mus, covs) for ws, mus, covs in zip(ws_list, mus_list, covs_list)])
    return np.argmax(log_probs, axis=0)

def eval(dev_subs_mean, ws_list, mus_list, covs_list):
    # Classify test images
    dev_true_labels = []
    dev_predicted_labels = []

    print(f"Evaluating GMM classes")
    for i in range(1, CLASSES + 1):
        dev_true_labels.extend([i - 1] * len(dev_subs_mean[i]))
        dev_predicted_labels.extend(classify_gmm(dev_subs_mean[i], ws_list, mus_list, covs_list))

    dev_true_labels = np.array(dev_true_labels)
    dev_predicted_labels = np.array(dev_predicted_labels)

    # Calculate accuracy
    accuracy = np.sum(dev_true_labels == dev_predicted_labels) / len(dev_true_labels)
    print("Accuracy:", accuracy)

# def predict_gmm(data):
#     mean_face = calculate_mean_face()

#     print(f"Creating subs mean class {i}")
#     V, S, U = np.linalg.svd(get_train_all(), full_matrices=False)

#     dev_subs_mean = (data - mean_face).dot(U.T)
#     result = classify_gmm(dev_subs_mean, ws_list, mus_list, covs_list)

#     return result


# dev_subs_mean, ws_list, mus_list, covs_list = train_gmm()
# eval(dev_subs_mean, ws_list, mus_list, covs_list)