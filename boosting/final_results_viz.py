import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

target_folder = "viz/optimal_viz"
plt.figure(figsize = (10,8))
i = 0
for file in ['results_Norrland_Hgv.jpg', 'results_Norrland_Dgv.jpg', 'results_Norrland_Volume.jpg',
             'results_Lettland_Hgv.jpg', 'results_Lettland_Dgv.jpg', 'results_Lettland_Volume.jpg',]:

    img = plt.imread(os.path.join(target_folder, file))
    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.axis('off')
    i += 1
plt.subplots_adjust(wspace=0.1, hspace=-0.5)
plt.savefig(os.path.join('viz/optimal_viz', 'final_viz.jpg'), dpi = 600, bbox_inches = 'tight', pad_inches = 0.1)