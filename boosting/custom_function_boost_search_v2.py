import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import tree
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import os

v1s = [0.005, 0.007, 0.01, 0.03, 0.07]
target_tree_size_list = [1, 2, 3, 4]
source_tree_size_list = [1, 2, 3, 4]
decay_factor_list = [0.97, 0.99, 0.993, 0.995, 0.997]
train_size_list = [0.1, 0.25, 0.5, 0.75, 1.0]
test_size_list = [0.25]
epoch_list = [400, 600]
alpha_0_list = [1.0]
test_seed_list = [1]
train_seed_list = [1]


def train_run(idx):
    v1_list = [v1s[int(idx)]]

    def create_train_test_split(test_size=0.25,
                                train_size=0.5,
                                target_col="Dgv",
                                use_sweden=True,
                                test_seed=1,
                                train_seed=1):

        #define predictor columns and target column
        predictor_columns = [
            'zq5', 'zq10', 'zq15', 'zq20', 'zq25', 'zq30', 'zq35', 'zq40',
            'zq45', 'zq50', 'zq55', 'zq60', 'zq65', 'zq70', 'zq75', 'zq80',
            'zq85', 'zq90', 'zq95', 'pzabovezmean', 'pzabove2', 'zpcum1',
            'zpcum2', 'zpcum3', 'zpcum4', 'zpcum5', 'zpcum6', 'zpcum7',
            'zpcum8', 'zpcum9'
        ]

        target_column = 'D_AVERAGE'

        data_sweden = pd.read_csv(
            'Image_data/merged_data_permanent_cleaned.csv', index_col=[0])

        #rename 'Hgv' to 'H_AVERAGE'
        data_sweden = data_sweden.rename(columns={target_col: "D_AVERAGE"})

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
            data_train_sweden,
            test_size=1 - train_size,
            random_state=train_seed
        )  #random state is not fixed, data_test unused

        if use_sweden == False:
            #evaluate and rain on latvia instead (keep naming for simplicity)
            data_latvia = pd.read_csv(
                'Latvian_Image_data/merged_data_cleaned.csv')
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

    #leaves returned by sklearn are not incremental by one. They can be for instance 1,3,4,7. This mapping should return (in this case) 0,1,2,3
    #this will be needed (I think) to create the matrices in a better way. We also need it to map test datapoints to correct leaves
    def map_leaves_to_number(leaves):
        unique_leaves = sorted(np.unique(leaves))
        mapped_values = np.arange(len(unique_leaves))
        mapping = []
        for leaf in leaves:
            index = np.where(unique_leaves == leaf)[0][0]
            mapping.append(mapped_values[index])

        mapping = np.array(mapping)
        return mapping

    def compute_rmse(predictions, targets):
        """Compute Root Mean Squared Error (RMSE)."""
        return np.sqrt(np.mean((predictions - targets)**2))

    def boosted_prediction(a_test, a_train, b_train, model_tray_clf,
                           model_tray_clfhat, leaf_gammas_tray,
                           leaf_gammashats_tray, v1, v2, alpha_tray):

        F_pred = np.full_like(
            a_test[:, 0], np.mean(b_train))  # Initialize with training mean

        # Iterate over each model (clf, clfhat) in the model tray
        for i, clf in enumerate(model_tray_clf):
            clfhat = model_tray_clfhat[i]
            alpha = alpha_tray[i]

            # Get leaf indices for both train and test. We need it as test datapoints are not necessarily contained in all regions unlike training datapoints
            leaves_clf_train = clf.apply(a_train)
            leaves_clfhat_train = clfhat.apply(a_train)

            leaves_clf_test = clf.apply(a_test)
            leaves_clfhat_test = clfhat.apply(a_test)

            leaves_clf_train_test = np.concatenate(
                (leaves_clf_train, leaves_clf_test))
            leaves_clfhat_train_test = np.concatenate(
                (leaves_clfhat_train, leaves_clfhat_test))

            indexed_leaves_clf_train_test = map_leaves_to_number(
                leaves_clf_train_test)
            indexed_leaves_clfhat_train_test = map_leaves_to_number(
                leaves_clfhat_train_test)

            #Now map the test datapoints to correct leaves
            indexed_leaves_clf_test = indexed_leaves_clf_train_test[
                len(leaves_clf_train):]

            indexed_leaves_clfhat_test = indexed_leaves_clfhat_train_test[
                len(leaves_clfhat_train):]

            # Retrieve gamma values
            leaf_gamma = leaf_gammas_tray[i]
            leaf_gammahat = leaf_gammashats_tray[i]
            #v2 -= 1/epochs

            for index in np.unique(indexed_leaves_clf_test):
                F_pred[indexed_leaves_clf_test ==
                       index] += v1 * leaf_gamma[index] * (1 - alpha)

            for index in np.unique(indexed_leaves_clfhat_test):

                F_pred[indexed_leaves_clfhat_test ==
                       index] += v2 * leaf_gammahat[index] * alpha

        return F_pred

    def LSTreeBoost(a,
                    b,
                    ahat,
                    bhat,
                    x_test,
                    y_test,
                    v1=0.1,
                    v2=0.1,
                    epochs=100,
                    target_tree_size=2,
                    source_tree_size=2,
                    alpha_0=1.0,
                    decay_factor=0.99,
                    eval=True):
        """LS TreeBoost with proper indexing for different domain sizes but a single function estimate F."""
        # Concatenate all unique data points
        all_X = np.concatenate((a, ahat))  # Combine target and source features
        F = np.full(all_X.shape[0],
                    np.mean(b))  # Initialize single function estimate
        F_source = np.full(all_X.shape[0], np.mean(bhat))
        all_y = np.concatenate((b, bhat))
        #F_source = F
        # Track trees and rho values
        model_tray_clf = []
        model_tray_clfhat = []
        leaf_gammas_tray = []
        leaf_gammashats_tray = []
        losses_target = []
        losses_source = []
        alpha_tray = []
        alpha = alpha_0
        losses_test = []
        epochs_test = []
        # Indices for each dataset within the combined set
        target_indices = np.arange(len(b))
        source_indices = np.arange(len(b), len(b) + len(bhat))
        all_indices = np.hstack([target_indices, source_indices])
        for m in range(epochs):
            # Compute residuals for the corresponding subsets
            b_residuals = b - F[target_indices]
            bhat_residuals = bhat - F[source_indices]
            b_residuals_all = all_y - F
            #b_residuals_all[source_indices] = bhat_residuals
            # Train decision trees on both target and source residuals
            clf = DecisionTreeRegressor(max_depth=target_tree_size,
                                        min_samples_leaf=25)
            clf.fit(a, b_residuals)

            clfhat = DecisionTreeRegressor(max_depth=source_tree_size,
                                           min_samples_leaf=25)
            clfhat.fit(ahat, bhat_residuals)

            model_tray_clf.append(clf)
            model_tray_clfhat.append(clfhat)

            # Get leaf indices for both train and test
            leaves_clf_target = clf.apply(a)
            leaves_clfhat_target = clfhat.apply(a)

            leaves_clf_source = clf.apply(ahat)
            leaves_clfhat_source = clfhat.apply(ahat)

            # Combine leaves from train and test for consistent indexing
            all_leaves_clf = np.concatenate(
                [leaves_clf_target, leaves_clf_source])
            all_leaves_clfhat = np.concatenate(
                [leaves_clfhat_target, leaves_clfhat_source])

            unique_leaves_clf = np.unique(all_leaves_clf)
            unique_leaves_clfhat = np.unique(all_leaves_clfhat)

            indexed_leaves_clf_train_test = map_leaves_to_number(
                all_leaves_clf)
            indexed_leaves_clfhat_train_test = map_leaves_to_number(
                all_leaves_clfhat)

            indexed_leaves_clf = indexed_leaves_clf_train_test[:len(
                leaves_clf_target)]

            indexed_leaves_clfhat = indexed_leaves_clfhat_train_test[:len(
                leaves_clfhat_target)]
            # print(f"Epoch {m}: unique_leaves_clfhat size = {len(unique_leaves_clfhat)}")
            #print(f"Epoch {m}: unique_leaves_clf size = {len(unique_leaves_clf)}")

            # Compute R matrix
            R = np.zeros((len(unique_leaves_clf), len(unique_leaves_clf)))
            for leaf_clf in indexed_leaves_clf:
                R[leaf_clf, leaf_clf] += 1

            # Compute Rhat matrix
            Rhat = np.zeros(
                (len(unique_leaves_clfhat), len(unique_leaves_clfhat)))
            for leaf_clfhat in indexed_leaves_clfhat:
                Rhat[leaf_clfhat, leaf_clfhat] += 1

            # Compute Intersection matrix N
            N = np.zeros((len(unique_leaves_clf), len(unique_leaves_clfhat)))
            for leaf_clf, leaf_clfhat in zip(indexed_leaves_clf,
                                             indexed_leaves_clfhat):
                N[leaf_clf, leaf_clfhat] += 1

            # Construct Block Matrix M
            upper = np.hstack((R, N))
            lower = np.hstack((N.T, Rhat))
            M = np.vstack((upper, lower))

            ### Build the residual vector r ###
            r1 = np.zeros(len(unique_leaves_clf))
            r2 = np.zeros(len(unique_leaves_clfhat))
            for index in sorted(np.unique(indexed_leaves_clf)):
                indixes = np.where(indexed_leaves_clf == index)[0]
                r1[index] = np.sum(b_residuals[indixes])
            for index in sorted(np.unique(indexed_leaves_clfhat)):
                indixes = np.where(indexed_leaves_clfhat == index)[0]
                r2[index] = np.sum(b_residuals[indixes])
            r = np.concatenate((r1, r2))
            #gamma_vector = r

            #print(f"Epoch {m}: r size = {len(r)}")
            #print(r)
            #print(f'residual vector: {r}')
            ### Solve to find gamma vector ###
            gamma_vector = np.linalg.lstsq(M, r, rcond=None)[0]
            #gamma_vector = np.linalg.solve(M, r) #for exact solution
            #extract gamma and gammahat separately
            leaf_gamma = gamma_vector[:len(unique_leaves_clf)]
            leaf_gammahat = gamma_vector[len(unique_leaves_clf):]

            #append to lists
            leaf_gammas_tray.append(leaf_gamma)
            leaf_gammashats_tray.append(leaf_gammahat)

            ###Below we can adjust alpha over time###
            alpha *= decay_factor
            alpha_tray.append(alpha)

            #Here we need this, since we should apply the update to all datapoints. So we need to consider all leaves
            indexed_leaves_clf = map_leaves_to_number(all_leaves_clf)
            indexed_leaves_clfhat = map_leaves_to_number(all_leaves_clfhat)
            #Update F
            for index in np.unique(indexed_leaves_clf):
                #print(len(F[np.where(indexed_leaves_clf == leaf)[0]]))
                F[indexed_leaves_clf ==
                  index] += v1 * leaf_gamma[index] * (1 - alpha)
                #F_target[indexed_leaves_clf == index] += v*leaf_gamma[index]
            for index in np.unique(indexed_leaves_clfhat):
                #print(len(F[np.where(indexed_leaves_clf == leaf)[0]]))
                F[indexed_leaves_clfhat ==
                  index] += v2 * leaf_gammahat[index] * alpha
                F_source[indexed_leaves_clfhat == index] += v2 * leaf_gammahat[
                    index]  #not used now. I add it so we can update residuals for source separately, if ever needed
                #F_source[indexed_leaves_clfhat == index] += v2*leaf_gammahat[index]

            # Compute RMSE loss on target domain
            mse_target = compute_rmse(F[target_indices], b)
            mse_source = compute_rmse(F[source_indices], bhat)

            #Evaluate on test set
            if m % 100 == 0:
                if eval:
                    preds = boosted_prediction(x_test, a, b, model_tray_clf,
                                               model_tray_clfhat,
                                               leaf_gammas_tray,
                                               leaf_gammashats_tray, v1, v2,
                                               alpha_tray)
                    mse_test = compute_rmse(preds, y_test)
                    losses_test.append(mse_test)
                    epochs_test.append(m)

            losses_target.append(mse_target)
            losses_source.append(mse_source)
            #if m % 100 == 0:
            #print(f'epoch: {m}')
            #print(f'alpha: {alpha}')
            # print(f'rmse: {mse_target}')
            # print(f'v1: {v1}')
            # print(f'v2: {v2}')
            # print(f'gamma_vec: {gamma_vector}')

        return losses_target, losses_source, losses_test, epochs_test, leaf_gammas_tray, leaf_gammashats_tray, model_tray_clf, model_tray_clfhat, alpha_tray

    df = pd.DataFrame(columns=[
        'id', 'train_seed', 'test_seed', 'train_size', 'test_size', 'v',
        'target_tree_size', 'source_tree_size', 'epochs', 'decay_factor',
        'alpha_0', 'test_rmse'
    ])

    i = 0
    for train_seed in train_seed_list:
        for test_seed in test_seed_list:
            for train_size in train_size_list:
                for test_size in test_size_list:
                    ahat_train, bhat_train, a_train, b_train, x_test, y_test = create_train_test_split(
                        test_size=test_size,
                        train_size=train_size,
                        target_col="Dgv",
                        use_sweden=True,
                        test_seed=test_seed,
                        train_seed=train_seed)
                    for v1 in v1_list:
                        v2 = v1
                        for target_tree_size in target_tree_size_list:
                            for source_tree_size in source_tree_size_list:
                                for epochs in epoch_list:
                                    for decay_factor in decay_factor_list:
                                        for alpha_0 in alpha_0_list:
                                            losses_target, losses_source, losses_test, epochs_test, leaf_gammas_tray, leaf_gammashats_tray, model_tray_clf, model_tray_clfhat, alpha_tray = LSTreeBoost(
                                                a_train,
                                                b_train,
                                                ahat_train,
                                                bhat_train,
                                                x_test,
                                                y_test,
                                                v1=v1,
                                                v2=v2,
                                                epochs=epochs,
                                                target_tree_size=
                                                target_tree_size,
                                                source_tree_size=
                                                source_tree_size,
                                                alpha_0=alpha_0,
                                                decay_factor=decay_factor,
                                                eval=True)
                                            preds = boosted_prediction(
                                                x_test,
                                                a_train,
                                                b_train,
                                                model_tray_clf,
                                                model_tray_clfhat,
                                                leaf_gammas_tray,
                                                leaf_gammashats_tray,
                                                v1=v1,
                                                v2=v2,
                                                alpha_tray=alpha_tray)
                                            test_rmse = compute_rmse(
                                                preds, y_test)
                                            df.loc[len(df)] = [
                                                int(i), train_seed, test_seed,
                                                len(a_train),
                                                len(x_test), v1,
                                                target_tree_size,
                                                source_tree_size, epochs,
                                                decay_factor, alpha_0,
                                                test_rmse
                                            ]
                                            df.to_csv(f'results_{v1}.csv')
                                            # Plot loss curve
                                            plt.plot(
                                                range(1,
                                                      len(losses_target) + 1),
                                                losses_target)
                                            plt.plot(
                                                range(1,
                                                      len(losses_source) + 1),
                                                losses_source)
                                            plt.plot(epochs_test, losses_test)
                                            plt.xlabel("Iteration")
                                            plt.ylabel("RMSE Loss")
                                            plt.legend([
                                                'target', 'source',
                                                'test (target)'
                                            ])
                                            plt.savefig(
                                                os.path.join(
                                                    'imgs',
                                                    str(int(i)) + '_' +
                                                    str(v1) + '.jpg'))
                                            plt.close('all')

                                            i += 1
