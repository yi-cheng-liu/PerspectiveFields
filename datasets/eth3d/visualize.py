import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob

# File paths
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
plt.savefig('dataset_distribution.png')