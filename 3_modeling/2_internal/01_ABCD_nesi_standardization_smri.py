# %%
#libraries
import pandas as pd
import numpy as np
import os
import sys
# import shutil
# import glob
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
# from nilearn import image as nli
# from nilearn import plotting
from joblib import Parallel, delayed
# from multiprocessing import Manager
from sklearn.cross_decomposition import PLSRegression
import gc
import traceback
import logging
from sklearn.model_selection import KFold
import random

# %%
## get fold as arg
if len(sys.argv) > 1:
    fold_num = int(sys.argv[1]) % 21

# %%
## set directory
root_dir = '/nesi/nobackup/uoo03493/farzane/abcd/'
std_dir = root_dir + 'main_std/'
if not os.path.isdir(std_dir):
    os.mkdir(std_dir) 

# %%
## load features and site and target tables
feature_set = 'smri_dict'
features = joblib.load(root_dir + feature_set + '.joblib')
demo = pd.read_csv(root_dir + 'demo_nesi.csv', index_col=0)
targs = pd.read_csv(root_dir + 'cog_all.csv', index_col=0, low_memory=False)
targs.index = targs.index.str.replace('_', '')


# %%
## load fold train and test indices created by kfold
Folds_inds = joblib.load(root_dir + 'Folds_inds.joblib')

# %%
def run_std(y, tr_index, te_index, path_out, target_name, flag):
    try:
        # create a dict to save standardized features and targets (be used as input in pls and enet , etc.)
        std_features = {}
        std_targets = {
            'train': {'std': None, 'raw': None},
            'test': {'std': None, 'raw': None}
        }
        # get train and test target
        y_tr = y.reindex(index = tr_index).dropna()
        #y_tr.to_csv(path_out + '/target_y_train.csv', header=False)
        y_te = y.reindex(index = te_index).dropna()
        #y_te.to_csv(path_out + '/target_y_test.csv', header=False)
        # fit and transform y std
        y_scaler = StandardScaler()
        y_tr_r = y_tr.values.reshape(-1, 1)
        y_tr_std = pd.DataFrame(y_scaler.fit_transform(y_tr_r),  index=y_tr.index)#columns=y1.columns,
        #y_tr_std.to_csv(path_out +'/removed_std_target_' + target_name + 'train.csv', header=False)
        #y_tr.to_csv(path_out+'/removed_target_' + target_name + '_train.csv', header=False)

        y_te_r = y_te.values.reshape(-1, 1)
        y_te_std = pd.DataFrame(y_scaler.transform(y_te_r),  index=y_te.index)#columns=y1.columns,
        #y_te_std.to_csv(path_out +'/removed_std_target_' + target_name + 'test.csv', header=False)
        #y_te.to_csv(path_out+'/removed_target_' + target_name + '_test.csv', header=False)

        std_targets['train']['std'] = y_tr_std
        std_targets['test']['std'] = y_te_std
        std_targets['train']['raw'] = y_tr
        std_targets['test']['raw'] = y_te

        joblib.dump(std_targets, path_out + '/' + target_name + 'std_targets.joblib', compress=1) 
        if flag:
            for key, feature in features.items():
                try:
                    x_scaler = StandardScaler()
                    x_scaler2 = StandardScaler()                
                    # get shared non na indices in  both target and feature
                    # trcommon_indices = feature.index.intersection(y_tr.index).dropna()
                    # tecommon_indices = feature.index.intersection(y_te.index).dropna()
                    # x_tr = feature.loc[trcommon_indices]
                    # y_tr = y_tr.loc[trcommon_indices]
                    # x_te = feature.loc[tecommon_indices]
                    # y_te = y_te.loc[tecommon_indices]
                    x_tr = feature.reindex(index = tr_index).dropna()
                    x_te = feature.reindex(index = te_index).dropna()

                    # fit and transform x std
                    x_tr_std = pd.DataFrame(x_scaler.fit_transform(x_tr), columns=x_tr.columns, index=x_tr.index)
                    #x_tr_std.to_csv(path_out + '/removed_' + str(key) + '_train.csv', index = True)

                    x_te_std = pd.DataFrame(x_scaler.transform(x_te), columns=x_te.columns, index=x_te.index)
                    #x_te_std.to_csv(path_out + '/removed_' + str(key) + '_test.csv', index = True)
                    x_te_std2 = pd.DataFrame(x_scaler2.fit_transform(x_te), columns=x_te.columns, index=x_te.index)
                    #add to dict
                    if key not in std_features:
                        std_features[key] = {}
                    std_features[key].update({'train': x_tr_std, 'test1': x_te_std, 'test2': x_te_std2}) 

                    gc.collect()
                except Exception as e:
                    logging.error(traceback.format_exc())
            # save dict
            joblib.dump(std_features, path_out + '/' + feature_set[0:4] + '_std_features.joblib', compress=1) 

        gc.collect()
    except Exception as e:
        logging.error(traceback.format_exc())

# %%

print('started to calculate the Fold #', fold_num, '\n')
# create directory for specific Fold
main_fold = std_dir + 'Fold_'+ str(fold_num)
if not os.path.isdir(main_fold):
    os.mkdir(main_fold) 
# get local indices for train and test
train_index = Folds_inds[fold_num][0]
test_index = Folds_inds[fold_num][1]
train_index = np.array(demo.iloc[train_index].index)
test_index = np.array(demo.iloc[test_index].index) 
# loop through targets
cog_cols = ['nihtbx_totalcomp_uncorrected', 'nihtbx_cryst_uncorrected', 'nihtbx_fluidcomp_uncorrected']
f_flag = True
for y_name in cog_cols:
    y = targs[y_name]
    target_name =  y_name.split('_')[1][0:5] + '_'
    run_std(y, train_index, test_index, main_fold, target_name, f_flag)
    f_flag = False



