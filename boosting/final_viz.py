import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os



VIZ_FOLDER = 'viz\optimal_viz'
RES_FOLDER = 'res\optimal_res'
TARGET_DATA = 'Lettland'
RESPONSE_VARIABLE = 'Volume'

if RESPONSE_VARIABLE == 'Dgv':
    label_to_use = 'Average diameter'
    unit = 'cm'
if RESPONSE_VARIABLE == 'Hgv':
    label_to_use = 'Average height'
    unit = 'm'
if RESPONSE_VARIABLE == 'Volume':
    label_to_use = 'Volume'
    unit = 'm$^3$ / ha'

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

    data['method'] = 'transfer'
    data_notransfer['method'] = 'no_transfer'
    columns = ['method', 'train_size', 'test_rmse']
    datavis = pd.concat([data_notransfer[columns], data[columns]])
    sns.pointplot(data=datavis, x='train_size', y='test_rmse', hue = 'method')
    plt.xlabel("train size")     # X-axis label
    plt.ylabel(f"RMSE ({unit})")  # Y-axis label
    plt.title(f" {label_to_use}, {TARGET_DATA}")    # Optional title
    plt.savefig(os.path.join(VIZ_FOLDER, f'results_{TARGET_DATA}_{RESPONSE_VARIABLE}.jpg'))
    plt.close('all')
    return None

data = combine_datasets(TARGET_DATA, RESPONSE_VARIABLE)
data_notransfer = read_notransfer_data(TARGET_DATA, RESPONSE_VARIABLE)
create_plots(data, data_notransfer)