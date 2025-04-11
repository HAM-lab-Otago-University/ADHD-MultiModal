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
#if len(sys.argv) > 1:
fold_num = int(sys.argv[1]) % 21


# %%
root_dir = '/nesi/nobackup/uoo03493/farzane/abcd/'
std_dir = root_dir + 'main_std/'
main_fold = std_dir+'Fold_'+str(fold_num) + '/'
enet1_dir = std_dir+'Fold_'+str(fold_num) + '/enet1/opadhd1/'
randfor_dir = main_fold + '/RF2/'
randfor2_dir = main_fold + '/RF2-2/'

out_dir = randfor_dir + 'opadhd1/'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir) 


te_dir = '/nesi/nobackup/uoo03493/farzane/abcd/opadhd1/'
te_std_dir = te_dir + 'main_std/opadhd_1/'

# %%
targ_list = ['total_','cryst_','fluid_'] #
te_targ_list = ['FSIQ', 'VIQ','PIQ']

# %%
contrasts3_v = ['cntr_twoback_task-verbal', 'cntr_oneback_task-verbal', 'cntr_twoback-oneback_task-verbal']

contrasts3_s = ['cntr_twoback_task-spatial', 'cntr_oneback_task-spatial', 'cntr_twoback-oneback_task-spatial']

# %%
modal_model = ['StrtFC', 'StrgFC', 'Str2', 'WmCntr2', 'All8', 'All9', 'All10']
# te_modal = ['cort', 'surf', 'subc', 'VolBrain', 'cntr_twobk-nback', 'cntr_onebk-nback', 'cntr_twobk-onebk-nback', 'verbal', 'spatial', 'tfc', 'gtfc']

moda_stack_dict = dict( StrtFC = ['cort', 'surf', 'subc', 'VolBrain', 'tfc'],
                        StrgFC = ['cort', 'surf', 'subc', 'VolBrain', 'tgfc'],
                        Str2 = ['cort', 'subc', 'surf', 'VolBrain'],
                        WmCntr2_v = contrasts3_v,
                        WmCntr2_s = contrasts3_s,
                        All8_v = ['tfc', 'cort', 'subc', 'surf', 'VolBrain'] + contrasts3_v,
                        All9_v = ['tgfc', 'cort', 'subc', 'surf', 'VolBrain'] + contrasts3_v,
                        All10_v = ['verbal', 'cort', 'subc', 'surf', 'VolBrain'] + contrasts3_v,
                        All8_s = ['tfc', 'cort', 'subc', 'surf', 'VolBrain'] + contrasts3_s,
                        All9_s = ['tgfc', 'cort', 'subc', 'surf', 'VolBrain'] + contrasts3_s,
                        All10_s = ['spatial', 'cort', 'subc', 'surf', 'VolBrain'] + contrasts3_s,
)


# %%
for i, targ in enumerate(targ_list):
    target = joblib.load(te_std_dir + te_targ_list[i] + '_std_targets.joblib')
    te_y = target['test']['std'].dropna()
    print('target size:', te_y.shape)
    # get test input (enet1):
    All_enet1 = joblib.load(enet1_dir + te_targ_list[i] + '_enet1_output_fstd.joblib')
    All_enet1['tgfc'] = All_enet1.pop('gtfc')
    print(All_enet1.keys())

    # load rf model :
    rf_model2 = joblib.load(randfor2_dir + targ + 'rf2-2_model_fstd.joblib')
    rf_model2_3 = joblib.load(randfor2_dir + targ + 'rf2-3_model_fstd.joblib')
    rf_model = {**rf_model2, **rf_model2_3}

    rf2_dict = {}

    for set_name, names in moda_stack_dict.items():
        print(set_name)
        for rf_set in modal_model:
            if rf_set in set_name:
                print('rf_set: ' , rf_set)
                set_model = rf_model[rf_set]['model']['best_model']
                print(type(set_model))
                set_dict = {key: All_enet1[key] for key in All_enet1.keys() if any(substring in key for substring in names)}
                print('set_dict: ', set_dict.keys())
                # Extract the indices from the DataFrames
                test_ind =  set()
                # Perform a full outer join on the indices
                set_df_te = None
                for moda_n , moda in set_dict.items():
                    print(moda['data']['yptest'].shape)
                    moda['data']['yptest'].columns = [f"{moda_n}" for col in moda['data']['yptest'].columns]
                    if set_df_te is None:
                        set_df_te = moda['data']['yptest']
                    else:
                        set_df_te = set_df_te.join(moda['data']['yptest'], how='outer', sort=True)
                    print(moda_n , set_df_te.shape)
                # # If there are duplicate columns due to the join, rename them
                # set_df_tr.columns = pd.io.parsers.ParserBase({'names':set_df_tr.columns})._maybe_dedup_names(set_df_tr.columns)
                # set_df_te.columns = pd.io.parsers.ParserBase({'names':set_df_te.columns})._maybe_dedup_names(set_df_te.columns)
                # set_df_tr.to_csv(enet2_dir + targ + set_name + '_df.csv')
                # Reorder columns
                if set_df_te is not None:
                    
                    set_df_te = set_df_te[sorted(set_df_te.columns)]
                    
                    #replace nan with extremes
                    te_l = set_df_te.fillna(1000)
                    te_s = set_df_te.fillna(-1000)
                    set_df_te = te_l.join(te_s, lsuffix='_l', rsuffix='_s')

                    # get test shared x and y indices
                    x_te = set_df_te.dropna()
                    y_te = target['test']['std'].dropna()
                    common_ind = list(set(y_te.index).intersection(set(x_te.index)))
                    x_te = x_te.loc[common_ind]
                    y_te = y_te.loc[common_ind]

                    # keys for output dict
                    if set_name not in rf2_dict:
                        rf2_dict[set_name] = {'data': {}}
                    print(targ)

                    # predict test y

                    te_y_p = set_model.predict(x_te.values)
                    te_y_rf2 = pd.DataFrame(te_y_p, index=y_te.index)
                    print(set_name, ' fit done', te_y_rf2.shape)
                    rf2_dict[set_name]['data'].update({'yptest': te_y_rf2, 'yttest': y_te}) 
    # save dict
    joblib.dump(rf2_dict, out_dir + te_targ_list[i] + '_rf2_output_fstd.joblib', compress=0)









