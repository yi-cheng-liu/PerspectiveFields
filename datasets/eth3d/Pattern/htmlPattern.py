from html4vision import Col, imagetable
import os
import glob
import re

base_folder = './'
cols = [
    Col('text', 'Pattern', ['Original', 'ZXY'])
]


for scene_folder in os.listdir(base_folder):
    scene_path = os.path.join(base_folder, scene_folder)
    if not os.path.isdir(scene_path):
        continue
    
    cols.extend([
        Col('img', scene_folder, f'{scene_folder}/*_pattern.png'),
        # Col('img', scene_folder, f'{scene_folder}/test_ZXY_pattern.png'),
    ])
    

imagetable(cols, imsize=(640, 480), sortable=True, out_file='index.html',
            sticky_header=True, sort_style='materialize', zebra=True)
