import os
import random
import numpy as np
from ikrlib import png2fea
from PIL import Image
import Augmentor

def augment_images(input_dir, output_dir, num_augmentations=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cls in range(1, 3):
        in_dir = input_dir + '/' + str(cls)
        for file_name in os.listdir(in_dir):
            out_dir = output_dir + '/' + str(cls)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            if file_name.endswith(".png"):
                p = Augmentor.Pipeline(source_directory=in_dir, output_directory=out_dir)
                p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
                p.zoom_random(probability=0.5, percentage_area=0.8)
                p.flip_left_right(probability=0.3)
                p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=1)
                p.skew_tilt(probability=0.5, magnitude=0.1)
                p.random_erasing(probability=0.2, rectangle_area=0.1)
                p.sample(num_augmentations)