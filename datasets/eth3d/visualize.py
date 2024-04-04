import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import os
from scipy.spatial.transform import Rotation as R

def visualize_distribution():
    # Output folder
    output_folder = 'Visualization'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Input files
    file_paths = glob.glob('*/test.json')

    combined_data = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            combined_data.extend(data['data'])
        
    df = pd.DataFrame(combined_data)

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribution of Roll, Pitch, Yaw, and VFOV')

    sns.histplot(df['roll'], bins=20, kde=True, ax=axs[0, 0])
    axs[0, 0].set_title('Roll Distribution')
    sns.histplot(df['pitch'], bins=20, kde=True, ax=axs[0, 1])
    axs[0, 1].set_title('Pitch Distribution')
    sns.histplot(df['yaw'], bins=20, kde=True, ax=axs[1, 0])
    axs[1, 0].set_title('Yaw Distribution')
    sns.histplot(df['vfov'], bins=20, kde=True, ax=axs[1, 1])
    axs[1, 1].set_title('VFOV Distribution')

    plt.tight_layout()
    plt.savefig(f'{output_folder}/dataset_distribution.png')

def pattern(file_paths):
    # Base output folder
    base_output_folder = 'Pattern/'
    
    for file_path in file_paths:
        # Scene name
        scene = os.path.basename(os.path.dirname(file_path))
        output_folder = os.path.join(base_output_folder, scene)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        folder_data = []
        
        with open(file_path, 'r') as file:
            data = json.load(file)
            folder_data.extend(data['data'])
            
            rolls = [item['roll'] for item in folder_data]
            pitches = [item['pitch'] for item in folder_data]
            yaws = [item['yaw'] for item in folder_data]
            image_ids = range(1, len(folder_data) + 1)

            # Plotting
            plt.figure(figsize=(10, 7))

            plt.plot(image_ids, rolls, label='Roll', marker='o')
            plt.plot(image_ids, pitches, label='Pitch', marker='x')
            plt.plot(image_ids, yaws, label='Yaw', marker='^')

            plt.title(f'{scene} Roll, Pitch, and Yaw')
            plt.xlabel('Image ID')
            plt.ylabel('Degrees')
            plt.legend()
            plt.grid(True)

            # Save each plot in the specific scene folder
            filename = os.path.basename(file_path).replace('.json', '_pattern.png')
            plt.savefig(os.path.join(output_folder, filename))
            print(f'Pattern plot saved to {output_folder}/{filename}')
            plt.close()

if __name__ == '__main__':
    file_paths = glob.glob('exhibition_hall/*.json')
    # file_paths = glob.glob('living_room/*.json')
    pattern(file_paths)