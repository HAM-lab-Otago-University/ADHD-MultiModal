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
#if len(sys.argv) > 1:
fold_num = int(sys.argv[1]) % 21

# %%
root_dir = '/nesi/nobackup/uoo03493/farzane/abcd/'
std_dir = root_dir + 'main_std/'
main_fold = std_dir+'Fold_'+str(fold_num) + '/'
pls_dir = main_fold + '/pls'
if not os.path.isdir(pls_dir):
    os.mkdir(pls_dir) 

# %%
feature_set = 'conn_std_features'

std_features = joblib.load(main_fold + feature_set + '.joblib')

# cog_cols = ['nihtbx_totalcomp_uncorrected', 'nihtbx_cryst_uncorrected', 'nihtbx_fluidcomp_uncorrected']
# y = targs['nihtbx_totalcomp_uncorrected']


# %%
def run_pls(path_out, target_name, pls_dict):
    try:
        # performance dict
        perf = {'nmse':[], 'r2':[], 'pr':[], 'mae':[]}
        # set params, use all components if less than 30
        nc = list(range(1, x_comp.shape[1]+1, 1)) if x_comp.shape[1] < 30 else list(range(1, 31, 1))
        param_grid = {'n_components': nc, 'scale': [True]}
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        # Create the PLS model
        pls = PLSRegression()
        # Perform grid search
        grid_search = GridSearchCV(pls, param_grid, scoring='neg_mean_squared_error', cv=KFold(5, shuffle = True, random_state=seed))
        grid_search.fit(x_comp.values, y_comp.values)
        # Get the best parameters
        best_model = grid_search.best_estimator_
        # save to disk
        # joblib.dump(grid_search.cv_results_, path_out + '/pls_cv_results_'+ target_name + str(key) + '.pkl')
        # joblib.dump(best_model, path_out + '/pls_best_model_' + target_name + str(key) + '.pkl')

        print(fold_num, 'set:' , key, ' nc:', grid_search.best_params_['n_components'], ' score:', grid_search.best_score_)

        # plot performance across folds
        n_components = grid_search.cv_results_['param_n_components'].data
        mean_test_scores = grid_search.cv_results_['mean_test_score']
        plt.plot(n_components, mean_test_scores, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Mean Test Score')
        plt.title(target_name + str(key) + '_Cross-validated Scores vs. Number of Components')
        plt.savefig(path_out + '/pls-score_' + target_name + str(key) +'.png')
        plt.close('all')

        # Transform the train using the PLS model
        tr_x_pls = pd.DataFrame(best_model.transform(x_comp.values), index=x_comp.index)
        # tr_x_pls.to_csv(path_out +'/' + target_name + str(key) + '_pls_train'+'.csv')
        # predict train y
        tr_y_p = best_model.predict(x_comp.values)
        tr_y_pls = pd.DataFrame(tr_y_p, index=y_comp.index)#.to_csv(path_out + '/predicted_y_'+ target_name + str(key) + '_pls_train'+'.csv')

        # add train performance 
        perf['nmse'].append(mean_squared_error(y_comp, tr_y_p))
        perf['r2'].append(r2_score(y_comp, tr_y_p))
        pearson_corr, _ = pearsonr(y_comp.values.flatten(), tr_y_p.flatten())
        perf['pr'].append(pearson_corr)
        perf['mae'].append(mean_absolute_error(y_comp, tr_y_p))


        #save to disk
        performance = pd.DataFrame(perf)

        # fill output dict
        if key not in pls_dict:
            pls_dict[key] = {'model': {}, 'data': {}}
        pls_dict[key]['data'].update({'Xtrain': tr_x_pls, 'yptrain': tr_y_pls, 'yttrain': y_comp}) 
        pls_dict[key]['model'].update({'perf': performance, 'cv_results': grid_search.cv_results_, 'best_model': best_model}) 
        gc.collect()
    except Exception as e:
        logging.error(traceback.format_exc())

# %%

print('started to calculate the Fold #', fold_num, '\n')

targ_list= ['total_','cryst_','fluid_'] #
for targ in targ_list:
    target = joblib.load(main_fold + targ + 'std_targets.joblib')
    # pls output dict for each target
    pls_dict = {}
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
        
        x_comp = pd.concat([x_tr, x_te])
        
        y_comp = pd.concat([y_tr, y_te])
                
        run_pls(pls_dir, targ, pls_dict)
    # save dict
    joblib.dump(pls_dict, pls_dir + '/' + targ + feature_set[0:4] + '_pls_output_fstd.joblib', compress=0) 
    gc.collect()    



