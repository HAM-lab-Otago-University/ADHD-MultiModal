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
from sklearn.ensemble import RandomForestRegressor
import shap

# %%
## get fold as arg

#if len(sys.argv) > 1:
fold_num = int(sys.argv[1]) % 21

# %%
root_dir = '/nesi/nobackup/uoo03493/farzane/abcd/'
std_dir = root_dir + 'main_std/'
main_fold = std_dir+'Fold_'+str(fold_num) + '/'
enet1_dir = std_dir+'Fold_'+str(fold_num) + '/enet1/'
pls1_dir = std_dir+'Fold_'+str(fold_num) + '/pls/'
randfor2_dir = main_fold + '/RF2/'
randfor2_2_dir = main_fold + '/RF2-2/'
randfor2_3_dir = main_fold + '/RF2-2/'
out_dir = pls1_dir + 'opadhd1/'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir) 

te_dir = '/nesi/nobackup/uoo03493/farzane/abcd/opadhd1/'
te_std_dir = te_dir + 'main_std/opadhd_1/'

# %%
targ_list = ['total_','cryst_','fluid_'] #
te_targ_list = ['FSIQ', 'VIQ','PIQ']


# %%

modal_model = ['cort', 'surf', 'subc', 'VolBrain', 'cntr_twobk_task-nback', 'cntr_zerobk_task-nback', 'cntr_twobk-zerobk_task-nback', 'cntr_twobk_task-nback', 'cntr_zerobk_task-nback', 'cntr_twobk-zerobk_task-nback', 'conn_wm', 'conn_wm', 'tfc', 'gfc']
te_modal = ['cort', 'surf', 'subc', 'VolBrain', 'cntr_twoback_task-verbal', 'cntr_oneback_task-verbal', 'cntr_twoback-oneback_task-verbal', 'cntr_twoback_task-spatial', 'cntr_oneback_task-spatial', 'cntr_twoback-oneback_task-spatial', 'verbal', 'spatial', 'tfc', 'tfc']

# %%

for i, targ in enumerate(targ_list):
    target = joblib.load(te_std_dir + te_targ_list[i] + '_std_targets.joblib')
    te_y = target['test']['std']
    print('target size:', te_y.shape, type(te_y))
    te_y =  te_y.dropna()
    pls1_dict = {}
    smri = joblib.load(pls1_dir + targ + 'smri_pls_output_fstd.joblib')
    cntr = joblib.load(pls1_dir + targ + 'cntr_pls_output_fstd.joblib')
    conn = joblib.load(pls1_dir + targ + 'conn_pls_output_fstd.joblib')
    gtfc = joblib.load(pls1_dir + targ + 'gtfc_pls_output_fstd.joblib') 
    All_enet1 = {**smri, **cntr, **conn, **gtfc} 

    te_smri = joblib.load(te_std_dir + 'smri_std_features.joblib')
    te_cntr = joblib.load(te_std_dir + 'cntr_std_features.joblib')
    te_conn = joblib.load(te_std_dir + 'conn_std_features.joblib')
    te_gtfc = joblib.load(te_std_dir + 'gtfc_std_features.joblib') 
    te_All_std = {**te_smri, **te_cntr, **te_conn, **te_gtfc} 

    for j, moda_n in enumerate(modal_model):
        if  moda_n in All_enet1.keys():
            print(moda_n, te_modal[j])
            en1_model = All_enet1[moda_n]['model']['best_model']
            te_x = te_All_std[te_modal[j]]['test1']
            te_x = te_x.dropna()
            print(te_modal[j], ' size:', te_x.shape)
            common_ind = list(set(te_x.index).intersection(set(te_y.index)))
            x_te = te_x.loc[common_ind]
            y_te = te_y.loc[common_ind]
            print(te_modal[j], 'shared size:', x_te.shape, y_te)
            te_x_pls = pd.DataFrame(en1_model.transform(x_te.values), index=x_te.index)
            te_y_p = en1_model.predict(x_te.values)
            print('model done', en1_model.n_components, te_x_pls.shape)

            te_y_eN1 = pd.DataFrame(te_y_p, index=x_te.index)
            if te_modal[j] not in pls1_dict:
                pls1_dict[te_modal[j]] = {'data': {}}
                pls1_dict[te_modal[j]]['data'].update({'Xtest': te_x_pls, 'yptest': te_y_eN1,  'yttest': y_te}) 
            else:
                pls1_dict['g'+te_modal[j]] = {'data': {}}
                pls1_dict['g'+te_modal[j]]['data'].update({'Xtest': te_x_pls, 'yptest': te_y_eN1,  'yttest': y_te}) 
                
    joblib.dump(pls1_dict, out_dir + te_targ_list[i] + '_pls1_output_fstd.joblib', compress=0)
    print(targ , 'done') 



