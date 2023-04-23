import os
import random
import numpy as np
from ikrlib import png2fea
from PIL import Image, ImageEnhance, ImageOps

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

def augment_images(input_dir, output_dir, num_augmentations=3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cls in range(1, 32):
        for file_name in os.listdir(input_dir + '/' + str(cls)):
            out_dir = output_dir + '/' + str(cls)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            if file_name.endswith(".png"):
                image_path = os.path.join(input_dir, str(cls), file_name)
                image = Image.open(image_path)

                for i in range(num_augmentations):
                    augmented_image = augment_image(image)
                    output_path = os.path.join(output_dir, str(cls), f"{file_name[:-4]}_aug_{i}.png")
                    augmented_image.save(output_path)
                    print(f"Data augumentation, new image created: {output_path}")