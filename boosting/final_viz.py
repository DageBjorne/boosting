import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os



VIZ_FOLDER = 'viz\optimal_viz'
RES_FOLDER = 'res\optimal_res'
TARGET_DATA = 'Norrland'
RESPONSE_VARIABLE = 'Dgv'

if RESPONSE_VARIABLE == 'Dgv':
    label_to_use = 'Stem diameter'
    unit = 'cm'
if RESPONSE_VARIABLE == 'Hgv':
    label_to_use = 'Tree height'
    unit = 'm'
if RESPONSE_VARIABLE == 'Volume':
    label_to_use = 'Stem volume'
    unit = 'm$^3$ / ha'

if TARGET_DATA == 'Norrland':
    title_name = 'N. Norrland'
else:
    title_name = 'Latvia'

def combine_datasets(TARGET_DATA, RESPONSE_VARIABLE):
    matching_files = glob.glob(os.path.join(RES_FOLDER, f'results_optim_{TARGET_DATA}_{RESPONSE_VARIABLE}*.csv'))
    print(matching_files)
    filtered_file = [f for f in matching_files if not f.endswith('notransfer.csv')][0]
    data = pd.read_csv(filtered_file, index_col = [0])
    return data

def read_notransfer_data(TARGET_DATA, RESPONSE_VARIABLE):
    matching_files = glob.glob(os.path.join(RES_FOLDER, f'results_optim_{TARGET_DATA}_{RESPONSE_VARIABLE}*.csv'))
    filtered_file = [f for f in matching_files if f.endswith('notransfer.csv')][0] 
    data = pd.read_csv(filtered_file, index_col = [0])
    return data


def create_plots(data, data_notransfer):

    data['method'] = 'L2TransferTreeBoost'
    data_notransfer['method'] = 'L2TreeBoost'
    columns = ['method', 'train_size', 'test_rmse']
    datavis = pd.concat([data_notransfer[columns], data[columns]])
    datavis['train_size'] = datavis['train_size'].astype(int)
    custom_palette = {"L2TreeBoost": "black", "L2TransferTreeBoost": "orange"}
    custom_markers = ["s", "^"]
    sns.pointplot(data=datavis, x='train_size', y='test_rmse', hue = 'method', 
                  palette=custom_palette, markers = custom_markers, dodge=True)
    
    custom_handles = [
    plt.Line2D([0], [0], marker='s', color='black', linestyle='-', markersize=10, label='L2TreeBoost'),
    plt.Line2D([0], [0], marker='^', color='orange', linestyle='-', markersize=10, label='L2TransferTreeBoost'),
]
    plt.xlabel("train size", fontsize = 14)     # X-axis label
    plt.ylabel(f"RMSE ({unit})", fontsize = 14)  # Y-axis label
    plt.title(f" {label_to_use}", fontsize = 14)    # Optional title
    plt.legend(fontsize=14)
    plt.legend(handles=custom_handles, fontsize = 14, title="Method", title_fontsize = 14)
    plt.savefig(os.path.join(VIZ_FOLDER, f'results_{TARGET_DATA}_{RESPONSE_VARIABLE}.jpg'), dpi = 300)
    plt.close('all')
    return None

data = combine_datasets(TARGET_DATA, RESPONSE_VARIABLE)
data_notransfer = read_notransfer_data(TARGET_DATA, RESPONSE_VARIABLE)
create_plots(data, data_notransfer)