import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os



VIZ_FOLDER = 'viz\optimal_viz'
RES_FOLDER = 'res\optimal_res'
TARGET_DATA = 'Lettland'
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
    import matplotlib.lines as mlines

    data['method'] = 'L2TransferTreeBoost'
    data_notransfer['method'] = 'L2TreeBoost'
    columns = ['method', 'train_size', 'test_rmse']
    datavis = pd.concat([data_notransfer[columns], data[columns]])
    datavis['train_size'] = datavis['train_size'].astype(int)

    # Compute mean, std, and 95% CI
    grouped = datavis.groupby(['method', 'train_size'])
    summary = grouped['test_rmse'].agg(['mean', 'std', 'count']).reset_index()
    summary['ci95'] = 1.96 * summary['std'] / np.sqrt(summary['count'])

    fig, ax = plt.subplots(figsize=(8, 6))

    method_styles = {
        'L2TreeBoost': {'color': 'black', 'marker': 's'},
        'L2TransferTreeBoost': {'color': 'orange', 'marker': '^'}
    }

    for method, style in method_styles.items():
        subset = summary[summary['method'] == method]
        ax.errorbar(
            subset['train_size'],
            subset['mean'],
            yerr=subset['ci95'],
            fmt=style['marker'] + '-',
            color=style['color'],
            label=method,
            capsize=7,
            elinewidth=2.5,
            markerfacecolor=style['color'],
            markersize=8
        )

    ax.set_xlabel("Train size", fontsize=24)
    ax.set_ylabel(f"RMSE ({unit})", fontsize=24)
    ax.set_title(f"{label_to_use}", fontsize=30, pad = 20)
    ax.tick_params(axis='both', which='major', labelsize=18) 
    ax.grid(True)

    handles = [
        mlines.Line2D([], [], color='black', marker='s', linestyle='-', label='L2TreeBoost', markersize = 10),
        mlines.Line2D([], [], color='orange', marker='^', linestyle='-', label='L2TransferTreeBoost', markersize = 10)
    ]
    ax.legend(handles=handles, title="Method", fontsize=18, title_fontsize=18)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_FOLDER, f'results_{TARGET_DATA}_{RESPONSE_VARIABLE}.jpg'), dpi=300)
    plt.close('all')


data = combine_datasets(TARGET_DATA, RESPONSE_VARIABLE)
data_notransfer = read_notransfer_data(TARGET_DATA, RESPONSE_VARIABLE)
create_plots(data, data_notransfer)