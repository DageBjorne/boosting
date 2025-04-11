import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os



VIZ_FOLDER = 'viz'
RES_FOLDER = 'res'
TARGET_DATA = 'Norrland'
RESPONSE_VARIABLE = 'Volume'

def combine_datasets(TARGET_DATA, RESPONSE_VARIABLE):

    matching_files = glob.glob(os.path.join(RES_FOLDER, f'results_{TARGET_DATA}_{RESPONSE_VARIABLE}_*.csv'))
    filtered_files = [f for f in matching_files if not f.endswith('notransfer.csv')]
    print(filtered_files)
    datas = []
    for file in filtered_files:
        data = pd.read_csv(file)
        #print(data)
        datas.append(data)
        combined = pd.concat(datas, ignore_index=True)


    return combined

def read_notransfer_data(TARGET_DATA, RESPONSE_VARIABLE):
    matching_files = glob.glob(os.path.join(RES_FOLDER, f'results_{TARGET_DATA}_{RESPONSE_VARIABLE}_*.csv'))
    filtered_file = [f for f in matching_files if  f.endswith('notransfer.csv')][0] 
    data = pd.read_csv(filtered_file, index_col = [0])
    return data


def create_box_plot(data, data_notransfer, top):

    datavis = pd.DataFrame(columns = ['method', 'rmse', 'train_size'])
    best_rows_transfer = pd.DataFrame()
    best_rows_notransfer = pd.DataFrame()
    for train_size in np.unique(data['train_size']):
        data_ = data[data['train_size'] == train_size]
        data_notransfer_ = data_notransfer[data_notransfer['train_size'] == train_size]
        sorted_rmse_data = sorted(data_['test_rmse'])[0:top]
        sorted_rmse_data_notransfer = sorted(data_notransfer_['test_rmse'])[0:top]
        best_row_transfer = data_[data_['test_rmse'] <= sorted_rmse_data[top-1]]
        best_row_notransfer = data_notransfer_[data_notransfer_['test_rmse'] <= sorted_rmse_data_notransfer[top-1]]
        best_rows_transfer = pd.concat([best_rows_transfer, best_row_transfer])
        best_rows_notransfer = pd.concat([best_rows_notransfer, best_row_notransfer])
        for index, row in best_row_transfer.iterrows():
            datavis.loc[len(datavis)] = ['transfer', row['test_rmse'], row['train_size']]
        for index, row in best_row_notransfer.iterrows():
            datavis.loc[len(datavis)] = ['no_transfer', row['test_rmse'], row['train_size']]
    best_rows_transfer.to_csv(os.path.join(VIZ_FOLDER, f'optimal_params_{TARGET_DATA}_{RESPONSE_VARIABLE}.csv'))
    best_rows_notransfer.to_csv(os.path.join(VIZ_FOLDER, f'optimal_params_{TARGET_DATA}_{RESPONSE_VARIABLE}_notransfer.csv'))
    sns.lineplot(data=datavis, x='train_size', y='rmse', hue = 'method')
    plt.savefig(os.path.join(VIZ_FOLDER, f'results_{TARGET_DATA}_{RESPONSE_VARIABLE}.jpg'))
    return None

data = combine_datasets(TARGET_DATA, RESPONSE_VARIABLE)
data_notransfer = read_notransfer_data(TARGET_DATA, RESPONSE_VARIABLE)
create_box_plot(data, data_notransfer, 10)