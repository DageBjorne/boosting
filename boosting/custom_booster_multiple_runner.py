# optim
#0,id,train_seed,test_seed,train_size,test_size,v,target_tree_size,source_tree_size,epochs,decay_factor,alpha_0,test_rmse,setting_id,rank,total_rank
# 57,29,29.0,1.0,1.0,127.0,424.0,0.07,1.0,3.0,600.0,0.997,1.0,4.06601426477246,670,58,203

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import tree
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import os

import config as c

#TARGET = 'N.Norrland'  #Lettland
#RESPONSE = 'Hgv'  #Dgv #Volume
RUNS = 50

#läs in tio bästa från motsvarande param-csv
#kör 10 körningar för varje parameter-uppsättning


def create_train_test_split(test_size=0.25,
                            train_size=0.5,
                            RESPONSE="Dgv",
                            TARGET='N.Norrland',
                            test_seed=1,
                            train_seed=1):

    #define predictor columns and target column
    predictor_columns = c.predictor_columns

    target_column = c.RESPONSE

    #data from Svedala
    data_sweden = pd.read_csv('data/merged_data_permanent_cleaned.csv',
                              index_col=[0])

    #rename 'Hgv' to 'H_AVERAGE'
    #data_sweden = data_sweden.rename(columns={RESPONSE: "D_AVERAGE"})

    #for training and testing on Sweden only
    data_train_sweden_1 = data_sweden[data_sweden['area_code'] == 4]
    data_train_sweden_2 = data_sweden[data_sweden['area_code'] == 2]
    data_train_sweden_3 = data_sweden[data_sweden['area_code'] == 3]
    data_train_sweden_4 = data_sweden[data_sweden['area_code'] == 1]

    #data_train_sweden_234 = data_train_sweden_4
    data_train_sweden_234 = pd.concat(
        [data_train_sweden_2, data_train_sweden_3])
    data_train_sweden_234 = pd.concat(
        [data_train_sweden_234, data_train_sweden_4])
    data_train_sweden, data_test_sweden_1 = train_test_split(
        data_train_sweden_1, test_size=test_size,
        random_state=test_seed)  #random_state is fixed

    data_train_sweden_1, data_test = train_test_split(
        data_train_sweden, test_size=1 - train_size,
        random_state=train_seed)  #random state is not fixed, data_test unused

    if TARGET == 'Lettland':
        #evaluate and rain on latvia instead (keep naming for simplicity)
        data_latvia = pd.read_csv('data/merged_data_cleaned_lettland.csv')
        data_train_sweden, data_test_sweden_1 = train_test_split(
            data_latvia, test_size=test_size,
            random_state=test_seed)  #random_state is fixed

        data_train_sweden_1, data_test = train_test_split(
            data_train_sweden,
            test_size=1 - train_size,
            random_state=train_seed
        )  #random state is not fixed, data_test unused

    #"General" base dataset (to use for transfer)
    ahat_train = np.array(data_train_sweden_234[predictor_columns])
    bhat_train = np.array(data_train_sweden_234[target_column])

    #Specific train and test set
    a_train = np.array(data_train_sweden_1[predictor_columns])
    b_train = np.array(data_train_sweden_1[target_column])

    x_test = np.array(data_test_sweden_1[predictor_columns])
    y_test = np.array(data_test_sweden_1[target_column])

    return ahat_train, bhat_train, a_train, b_train, x_test, y_test


def compute_rmse(predictions, targets):
    """Compute Root Mean Squared Error (RMSE)."""
    return np.sqrt(np.mean((predictions - targets) ** 2))

# **Final Prediction using All Trees**
def boosted_prediction(X_new, b, model_tray, leaf_means_tray, v=0.1):
    """Compute the final LAD TreeBoost prediction."""
    F_pred = np.full_like(X_new[:, 0], np.mean(b))  # Start with initial median
    
    for i, tree in enumerate(model_tray):
        leaf_nodes = tree.apply(X_new)  # Find leaf for each X_new
        # Add the median residual for the region (leaf) of each X_new
        F_pred += v*np.array([leaf_means_tray[i][leaf] for leaf in leaf_nodes])
    
    return F_pred

 
def LSTreeBoost(a, b, x_test, y_test, epochs = 100, v = 0.1, tree_size = 2, eval=True):
    # Initialize the model with the median of y
    F = np.full_like(b, np.mean(b))  # F_0(x) = median(y)
    model_tray = []  # Store trees
    leaf_means_tray = []  # Store per-leaf median residuals
    losses = []
    losses_test = []
    epochs_test = []

    # Perform boosting iterations
    for m in range(epochs):
        
        # Compute residuals: r_i = y_i - F(x_i)
        b_residuals = (b - F)

        # Train a decision tree on the residuals
        clf = DecisionTreeRegressor(max_depth=tree_size, min_samples_leaf = 25)
        clf.fit(a, b_residuals)
        model_tray.append(clf)  # Store trained tree

        # Get the leaf node assignments for each data point
        leaf_nodes = clf.apply(a)



        # Compute median residual for each leaf
        unique_leaves = np.unique(leaf_nodes)
        leaf_means = {
            leaf: np.mean(b_residuals[leaf_nodes == leaf])  
            for leaf in unique_leaves
        }
        leaf_means_tray.append(leaf_means)  # Store median updates
        #print(leaf_means)
        # Update F by adding the median residual of the corresponding leaf for each sample
        for leaf in unique_leaves:
            F[leaf_nodes == leaf] += v*leaf_means[leaf]  # Update the samples in that region
        mse = compute_rmse(F, b)  # LSTreeBoost minimizes MAE
        losses.append(mse)
        if m % 100 == 0:
            if eval:
                preds = boosted_prediction(x_test, b, model_tray, leaf_means_tray, v=v)
                mse_test = compute_rmse(preds, y_test)
                losses_test.append(mse_test)
                epochs_test.append(m)
        
        
        
    return model_tray, leaf_means_tray, losses, losses_test, epochs_test






df = pd.DataFrame(columns=[
    'id', 'train_seed', 'test_seed', 'train_size', 'test_size', 'v',
    'tree_size', 'epochs', 'test_rmse'
])

param_list = pd.read_csv(
    f"res/optimal_params/optimal_params_{c.TARGET}_{c.RESPONSE}_notransfer.csv")

cols = [
    'train_size', 'v', 'tree_size', 'epochs',
    'total_rank', 'rank'
]
param_list = param_list[cols].drop_duplicates()
param_list = param_list.sort_values(by='total_rank',
                                    ascending=True).reset_index()

# loop over the tio besten paramsinsettlings!!!

#for j in range(10):
# för varje train_size välj bästa!!!

test_size = 0.25

j = 0
for t in c.train_size_list:
    train_size = t  #(t * 1.0) / 20

    p_list = param_list[param_list['train_size'] ==
                        np.unique(param_list['train_size'])[j]]  ## train_size in percetntae fix!!!
    j += 1
    p_list = p_list.sort_values(by='rank', ascending=True).reset_index()
    #print(p_list)
    v = p_list['v'][0]
    epochs = int(p_list['epochs'][0])
    tree_size = int(p_list['tree_size'][0])


    for i in range(RUNS):
        ahat_train, bhat_train, a_train, b_train, x_test, y_test = create_train_test_split(
            test_size=test_size,
            train_size=train_size,
            RESPONSE=c.RESPONSE,
            TARGET=c.TARGET,
            test_seed=i,
            train_seed=i + RUNS)

        model_tray, leaf_means_tray, losses, losses_test, epochs_test = LSTreeBoost(a_train, b_train, x_test, y_test, v=v,
                                                                                                        tree_size=tree_size,
                                                                                                        epochs = epochs)
        preds = boosted_prediction(x_test, b_train, model_tray, leaf_means_tray, v=v)
  
        test_rmse = compute_rmse(preds, y_test)
        df.loc[len(df)] = [
            int(i), i, i + RUNS,
            len(a_train),
            len(x_test), v, tree_size, epochs,
            test_rmse
        ]
        df.to_csv(f'res/optimal_res/results_optim_{c.TARGET}_{c.RESPONSE}_notransfer.csv')
