from html4vision import Col, imagetable
import os

image_base_dir = 'datasets/eth3d'
output_base_dir = 'output/eth3d'
subdirectories = [d for d in os.listdir(image_base_dir) if os.path.isdir(os.path.join(image_base_dir, d))]

cols = [
    Col('id1', 'ID'),
]

for subdir in subdirectories:
    resized_image_path = os.path.join(image_base_dir, subdir, 'images/dslr_images_resized/*.JPG')
    param_pred_path = os.path.join(output_base_dir, subdir, '*/param_pred.png')
    perspective_pred_path = os.path.join(output_base_dir, subdir, '*/perspective_pred.png')

    cols.append(Col('img', f'{subdir}', resized_image_path))
    cols.append(Col('img', f'{subdir} param prediction', param_pred_path))
    cols.append(Col('img', f'{subdir} perspective prediction', perspective_pred_path))

# Generate the HTML table
imagetable(cols, imsize=(640, 480), sortable=True, sticky_header=True, sort_style='materialize', zebra=True)
