# %%


# %%
#libraries
import pandas as pd
import numpy as np
import os
import sys
import joblib
import warnings
# from datetime import date, datetime
import pickle
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
import scipy.stats as st
from scipy.stats import zscore as  zscore
from joblib import Parallel, delayed
# from multiprocessing import Manager
from sklearn.cross_decomposition import PLSRegression
import gc
import traceback
import logging
from sklearn.model_selection import KFold
import random

# %%
if len(sys.argv) > 1:
    fold_num = int(sys.argv[1]) % 21

# %%
root_dir = '/nesi/nobackup/uoo03493/farzane/abcd/'
std_dir = root_dir + 'main_std/'
main_fold = std_dir+'Fold_'+str(fold_num) + '/'
enet1_dir = main_fold + '/enet1'
if not os.path.isdir(enet1_dir):
    os.mkdir(enet1_dir) 

# %%
feature_set = 'cntr_std_features'     # changes for each modality dict

std_features = joblib.load(main_fold + feature_set + '.joblib')

# %%
def run_enet1(path_out, target_name):
    try:

        perf = {'nmse':[], 'r2':[], 'pr':[], 'mae':[]}

        eNet_hyper_param = {'alpha': np.logspace(0, 2, 70), #70
                    'l1_ratio':np.linspace(0,1,25), #25
                    'max_iter': [1000], # 1
                    'tol': [1e-4],
                }
        reg = ElasticNet()        
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        search = GridSearchCV(reg, eNet_hyper_param, cv=5, scoring='neg_mean_squared_error', n_jobs = 50, verbose = 0)

        search.fit(x_tr.values, y_tr.values)

        best_model = search.best_estimator_
        if best_model is not None:
        #     joblib.dump(search.cv_results_, path_out + 'enet-l1_cv_results_'+ target_name + str(key)+'.pkl')
        #     joblib.dump(best_model, path_out + 'enet-l1_best_model_'+ target_name +str(key)+'.pkl')
            print('set:' , key , ' score:', search.best_score_)
            cv_results = search.cv_results_
            #print("Best Model Parameters:")
            #print(search.best_params_)
            # Extract relevant information
            params = cv_results['params']
            mean_test_scores = cv_results['mean_test_score'].reshape(len(eNet_hyper_param['alpha']), len(eNet_hyper_param['l1_ratio']))
            alpha_values = np.unique([param['alpha'] for param in params])
            l1_ratio_values = np.unique([param['l1_ratio'] for param in params])

            # Create a DataFrame for the heatmap
            df_heatmap = pd.DataFrame(mean_test_scores, index=alpha_values, columns=l1_ratio_values)
        
            #print('alpha:', len(alpha_values))
            #print('l1_ratio' , len(l1_ratio_values))
            #print('mtest_score' , len(mean_test_scores))
            # df_heatmap.to_csv(path_out + 'enet-l1_' + target_name + str(key)+'.csv')
            # Plot the heatmap
            # Plot the heatmap
            # plt.figure(figsize=(12, 8))
            # sns.heatmap(df_heatmap, annot=False, cmap='viridis', cbar_kws={'label': 'Mean Test Score'})
            # plt.title('Hyperparameter Heatmap')
            # plt.xlabel('Alpha')
            # plt.ylabel('L1 Ratio')
            # plt.savefig(path_out + '/enet-l1_' + target_name + str(key) + '.png')
            # plt.close('all')
            # predict train y
            tr_y_p = best_model.predict(x_tr.values)
            tr_y_eN1 = pd.DataFrame(tr_y_p, index=y_tr.index)

            # add train performance 
            perf['nmse'].append(mean_squared_error(y_tr, tr_y_p))
            perf['r2'].append(r2_score(y_tr, tr_y_p))
            pearson_corr, _ = pearsonr(y_tr.values.flatten(), tr_y_p.flatten())
            perf['pr'].append(pearson_corr)
            perf['mae'].append(mean_absolute_error(y_tr, tr_y_p))

            # predict test y
            te_y_p = best_model.predict(x_te.values)
            te_y_eN1 = pd.DataFrame(te_y_p, index=y_te.index)

            # add test performance 
            perf['nmse'].append(mean_squared_error(y_te, te_y_p))
            perf['r2'].append(r2_score(y_te, te_y_p))
            p_corr, _ = pearsonr(y_te.values.flatten(), te_y_p.flatten())
            perf['pr'].append(p_corr)
            perf['mae'].append(mean_absolute_error(y_te, te_y_p))

            #save to disk
            performance = pd.DataFrame(perf)
            # fill output dict
            enet1_dict[key]['data'].update({'yptrain': tr_y_eN1, 'yptest': te_y_eN1, 'yttrain': y_tr, 'yttest': y_te}) 
            enet1_dict[key]['model'].update({'perf': performance, 'cv_results': search.cv_results_, 'best_model': best_model, 'heatmap': df_heatmap}) 
        else:
            print('best_model is none')
            # plt.close('all')
        return best_model
    except Exception as e:
        logging.error(traceback.format_exc())

# %%

print('started to calculate the Fold #', fold_num, '\n')

targ_list = ['total_','cryst_','fluid_']
for targ in targ_list:
    target = joblib.load(main_fold + targ + 'std_targets.joblib')
    # enet output dict for each target
    enet1_dict = {}
    for key, feature in std_features.items():
        # get train shared x and y indices
        x_tr = feature['train'].dropna()
        y_tr = target['train']['std'].dropna()
        common_ind = list(set(y_tr.index).intersection(set(x_tr.index)))
        x_tr = x_tr.loc[common_ind]
        y_tr = y_tr.loc[common_ind]
        # get test shared x and y indices
        x_te = feature['test1'].dropna()
        y_te = target['test']['std']
        common_ind = list(set(y_te.index).intersection(set(x_te.index)))
        x_te = x_te.loc[common_ind]
        y_te = y_te.loc[common_ind]

        # keys for output dict
        if key not in enet1_dict:
            enet1_dict[key] = {'model': {}, 'data': {}}
        run_enet1(enet1_dir, targ)
    # save dict
    joblib.dump(enet1_dict, enet1_dir + '/' + targ + feature_set[0:4] + '_enet1_output_std.joblib', compress=0) 
        



