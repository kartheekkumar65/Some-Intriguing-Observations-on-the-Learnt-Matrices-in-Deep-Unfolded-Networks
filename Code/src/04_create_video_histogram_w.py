import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import imageio
import os
from tqdm import tqdm

import scienceplots


plt.style.use(['science', 'ieee'])
plt.rcParams.update({"font.family": "sans", "font.serif": [
                    "cm"], "mathtext.fontset": "cm", "font.size": 22})


W_list = np.load(f'./data/W_constant.npy')
time_list = np.load('./data/time_list_constant.npy')
model = "constant"

output_dir = './videos/histogram_videos_3fps'
os.makedirs(output_dir, exist_ok=True)

# Determine global min and max for X-axis to ensure consistent plotting
xmin = np.min([np.min(s) for s in W_list]) - 0.1
xmax = np.max([np.max(s) for s in W_list]) + 0.1


filenames = []

for i in tqdm(range(len(W_list))):
    plt.figure(figsize=(12, 6))
    
    sns.histplot(W_list[i].reshape(-1), bins=128, kde=False, color='salmon', stat='density', linewidth=0)
    
    mu, std = stats.norm.fit(W_list[i].reshape(-1))
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=1, linestyle='--')
    str_epoch = f'{i}'.zfill(2)
    str_time =  f'{round(float(time_list[i]), 4):0>8.3f}'
    plt.title(f"Epoch:{str_epoch}, Time:{str_time} seconds\nFIT RESULTS: $\mu$ = {mu:.2f}, $\sigma$ = {std:.2f}")
    plt.xlabel('VALUE')
    plt.ylabel('DENSITY')
    plt.grid(True)
    plt.ylim(0, 17)
    
    # Save each frame as a temporary PNG file
    filename = f'{output_dir}/W_{model}_ep_{i}.png'
    plt.savefig(filename)
    plt.close()
    
    filenames.append(filename)

# Create the GIF
with imageio.get_writer(f'{output_dir}/W_{model}_histograms.mp4', fps=3) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    

print("GIF creation completed.")
