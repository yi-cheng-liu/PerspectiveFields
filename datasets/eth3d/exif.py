from PIL import Image
from PIL.ExifTags import TAGS
import json
import os
from glob import glob

image_directory = '*/images/dslr_images'
image_files = glob(os.path.join(image_directory, '*.JPG'))

data = []

for image_path in image_files:
    with Image.open(image_path) as img:
        exif_data = img._getexif()
        for key, value in exif_data.items():
            if key in TAGS:
                data.append({TAGS[key] : value})


# Save the JSON data to a file
# json_file_path = '/path/to/save/exif_data.json'  # Adjust the path as necessary
# with open(json_file_path, 'w') as json_file:
#     json_file.write(json_data)

# print(f"EXIF data saved to {json_file_path}")