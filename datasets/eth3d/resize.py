from PIL import Image
import os
from glob import glob

base = os.getcwd()
subdirectories = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]

new_size = (640, 480)
for folder_name in subdirectories:
    image_path = os.path.join(folder_name, 'images/dslr_images')
    target_path = os.path.join(folder_name, 'images/dslr_images_resized')

    if os.path.exists(image_path):  # Ensure the source directory exists before proceeding
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        image_files = glob(os.path.join(image_path, '*.JPG'))

        for image_path in image_files:
            with Image.open(image_path) as img:
                img_resized = img.resize(new_size, Image.ANTIALIAS)
                resized_image_path = os.path.join(target_path, os.path.basename(image_path))
                img_resized.save(resized_image_path)

        print(f"Images have been resized and saved to {target_path}")