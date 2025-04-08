import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

VIZ_FOLDER = 'viz'
RES_FOLDER = 'res'
TARGET_DATA = 'Norrland'
RESPONSE_VARIABLE = 'Dgv'

def combine_datasets(TARGET_DATA, RESPONSE_VARIABLE):

    matching_files = glob.glob(os.path.join(RES_FOLDER, f'results_{TARGET_DATA}_{RESPONSE_VARIABLE}_*.csv'))
    datas = []
    for file in matching_files:
        data = pd.read_csv(file, index_col = [0])
        datas.append(data)

    data_ = datas[0]
    for data in datas[1:]:
        data_ = pd.concat([data_, data])
    return data_

def create_box_plot(data, top):

    datavis = pd.DataFrame(columns = ['method', 'rmse', 'train_size'])

    for train_size in np.unique(data['train_size']):
        data_ = data[data['train_size'] == train_size]
        sorted_rmse_data = sorted(data_['test_rmse'])[0:top]
        best_row_transfer = data_[data_['test_rmse'] <= sorted_rmse_data[top-1]]

        for index, row in best_row_transfer.iterrows():
            datavis.loc[len(datavis)] = ['transfer', row['test_rmse'], row['train_size']]
    sns.boxplot(data=datavis, x='train_size', y='rmse')
    plt.savefig(os.path.join(VIZ_FOLDER, f'results_{TARGET_DATA}_{RESPONSE_VARIABLE}.jpg'))
    return None

data = combine_datasets(TARGET_DATA, RESPONSE_VARIABLE)
create_box_plot(data, 5)