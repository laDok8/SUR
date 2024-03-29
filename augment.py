import os
import Augmentor

def augment_images(input_dir, output_dir, num_augmentations=10, random_noice=False, num_classes=31):
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            for cls in range(1, num_classes + 1):
                if os.path.exists(os.path.join(output_dir, str(cls))):
                    for f in os.listdir(os.path.join(output_dir, str(cls))):
                        os.remove(os.path.join(output_dir, str(cls), f))
                    os.rmdir(os.path.join(output_dir, str(cls)))
        os.rmdir(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cls in range(1, num_classes + 1):
        in_dir = os.path.join(input_dir,str(cls))
        out_dir = os.path.join(output_dir,str(cls))
        if os.path.exists(os.path.join(in_dir,'out')):
            for f in os.listdir(os.path.join(in_dir,'out')):
                os.remove(os.path.join(in_dir,'out',f))
            os.rmdir(os.path.join(in_dir,'out'))
        if not os.path.join(output_dir):
            os.makedirs(output_dir)
        p = Augmentor.Pipeline(source_directory=in_dir, output_directory='out')
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        p.zoom_random(probability=0.5, percentage_area=0.8)
        p.flip_left_right(probability=0.3)
        p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=1)
        p.skew_tilt(probability=0.5, magnitude=0.1)
        if random_noice:
            p.random_erasing(probability=0.2, rectangle_area=0.1)
        p.sample(num_augmentations)
        os.rename(os.path.join(in_dir,'out'), out_dir)