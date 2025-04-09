import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os



VIZ_FOLDER = 'viz'
RES_FOLDER = 'res'
TARGET_DATA = 'Norrland'
RESPONSE_VARIABLE = 'Hgv'

def combine_datasets(TARGET_DATA, RESPONSE_VARIABLE):

    matching_files = glob.glob(os.path.join(RES_FOLDER, f'results_{TARGET_DATA}_{RESPONSE_VARIABLE}_*.csv'))
    filtered_files = [f for f in matching_files if not f.endswith('notransfer.csv')]
    datas = []
    for file in filtered_files:
        data = pd.read_csv(file, index_col = [0])
        datas.append(data)

    data_ = datas[0]
    for data in datas[1:]:
        data_ = pd.concat([data_, data])
    return data_

def read_notransfer_data(TARGET_DATA, RESPONSE_VARIABLE):
    matching_files = glob.glob(os.path.join(RES_FOLDER, f'results_{TARGET_DATA}_{RESPONSE_VARIABLE}_*.csv'))
    filtered_file = [f for f in matching_files if  f.endswith('notransfer.csv')][0] 
    data = pd.read_csv(filtered_file, index_col = [0])
    return data


def create_box_plot(data, data_notransfer, top):

    datavis = pd.DataFrame(columns = ['method', 'rmse', 'train_size'])

    for train_size in np.unique(data['train_size']):
        data_ = data[data['train_size'] == train_size]
        data_notransfer_ = data_notransfer[data_notransfer['train_size'] == train_size]
        sorted_rmse_data = sorted(data_['test_rmse'])[0:top]
        sorted_rmse_data_notransfer = sorted(data_notransfer_['test_rmse'])[0:top]
        best_row_transfer = data_[data_['test_rmse'] <= sorted_rmse_data[top-1]]
        best_row_notransfer = data_notransfer_[data_notransfer_['test_rmse'] <= sorted_rmse_data_notransfer[top-1]]

        for index, row in best_row_transfer.iterrows():
            datavis.loc[len(datavis)] = ['transfer', row['test_rmse'], row['train_size']]
        for index, row in best_row_notransfer.iterrows():
            datavis.loc[len(datavis)] = ['no_transfer', row['test_rmse'], row['train_size']]
    sns.lineplot(data=datavis, x='train_size', y='rmse', hue = 'method')
    plt.savefig(os.path.join(VIZ_FOLDER, f'results_{TARGET_DATA}_{RESPONSE_VARIABLE}.jpg'))
    return None

data = combine_datasets(TARGET_DATA, RESPONSE_VARIABLE)
data_notransfer = read_notransfer_data(TARGET_DATA, RESPONSE_VARIABLE)
create_box_plot(data, data_notransfer, 10)