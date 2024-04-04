from html4vision import Col, imagetable
import os
import json

image_base_dir = 'datasets/eth3d'
output_base_dir = 'output/eth3d'
output_base_dir_2 = 'output2/eth3d'
ignored_folders = ['Pattern', 'observatory', 'door', 'statue']#, 'botanical_garden', 'boulders', 'bridge', 'lecture_room', 'lounge', 'old_computer', 'terrace_2', 'living_room']
subdirectories = [d for d in os.listdir(image_base_dir) if os.path.isdir(os.path.join(image_base_dir, d)) and d not in ignored_folders]
subdirectories = sorted(subdirectories)

cols = [
    Col('id1', 'ID'),
]

for subdir in subdirectories:
    resized_image_path = os.path.join(image_base_dir, subdir, 
                                      'images/dslr_images_resized/*.JPG')
    param_pred_path = os.path.join(output_base_dir, subdir, 
                                   '*/param_pred.png')
    param_pred_path_2 = os.path.join(output_base_dir_2, subdir, 
                                     '*/param_pred.png')
    perspective_pred_path = os.path.join(output_base_dir, subdir, 
                                         '*/perspective_pred.png')
    json_path = os.path.join(image_base_dir, subdir, 'test_ZXY.json')
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        name = []
        rolls, pitchs, yaws, vfovs = [], [], [], []
        for item in json_data['data']:
            name.append(item['file_name'])
            rolls.append(round(item['roll'], 5))
            pitchs.append(round(item['pitch'], 5))
            yaws.append(round(item['yaw'], 5))
            vfovs.append(round(item['vfov'], 5))
            
    print(param_pred_path_2)

    cols.extend([
        # Col('text', f'{subdir} Name', name),
        Col('img', f'{subdir}', resized_image_path),
        Col('img', f'{subdir} param prediction', param_pred_path),
        Col('img', f'{subdir} Ground Truth param prediction', param_pred_path_2),
        # Col('img', f'{subdir} perspective prediction', perspective_pred_path),
        Col('text', f'{subdir} Roll', rolls),
        Col('text', f'{subdir} Pitch', pitchs),
        Col('text', f'{subdir} Yaw', yaws),
        Col('text', f'{subdir} VFOV', vfovs),
        # Col('text', f'{subdir} Original Roll', original_rolls),
        # Col('text', f'{subdir} Original Pitch', original_pitchs),
        # Col('text', f'{subdir} Original VFOV', original_vfovs)
    ])

# Generate the HTML table
imagetable(cols, imsize=(320, 240), sortable=True, sticky_header=True, 
           sort_style='materialize', zebra=True)
