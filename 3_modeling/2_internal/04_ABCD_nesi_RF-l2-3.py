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
if len(sys.argv) > 1:
    fold_num = int(sys.argv[1]) % 21


# %%
root_dir = '/nesi/nobackup/uoo03493/farzane/abcd/'
std_dir = root_dir + 'main_std/'
main_fold = std_dir+'Fold_'+str(fold_num) + '/'
enet1_dir = std_dir+'Fold_'+str(fold_num) + '/enet1/'
randfor2_dir = main_fold + '/RF2-2/'
if not os.path.isdir(randfor2_dir):
    os.mkdir(randfor2_dir) 



# %%
def run_RF2(path_out, target_name):
    try:

        perf = {'nmse':[], 'r2':[], 'pr':[], 'mae':[]}
        rf_hyper_param = {
            'n_estimators': [1000],  # Number of trees in the forest
            'max_depth': list(range(1,11)),  # Maximum depth of the tree
            'max_features': [None, 'sqrt', 'log2'] 
        }
        reg = RandomForestRegressor()        
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        search = GridSearchCV(reg, rf_hyper_param, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
        search.fit(x_tr.values, y_tr.values)
        print(set_name, ' grd search done')
        best_model = search.best_estimator_
        if best_model is not None:
        #     joblib.dump(search.cv_results_, path_out + 'enet-l1_cv_results_'+ target_name + str(key)+'.pkl')
        #     joblib.dump(best_model, path_out + 'enet-l1_best_model_'+ target_name +str(key)+'.pkl')
            print('set:' , set_name , ' score:', search.best_score_)
            cv_results = search.cv_results_
            df_heatmap = pd.DataFrame(cv_results)

            # predict train y
            tr_y_p = best_model.predict(x_tr.values)
            tr_y_rf2 = pd.DataFrame(tr_y_p, index=y_tr.index)

            # add train performance 
            perf['nmse'].append(mean_squared_error(y_tr, tr_y_p))
            perf['r2'].append(r2_score(y_tr, tr_y_p))
            pearson_corr, _ = pearsonr(y_tr.values.flatten(), tr_y_p.flatten())
            perf['pr'].append(pearson_corr)
            perf['mae'].append(mean_absolute_error(y_tr, tr_y_p))

            # predict test y
            te_y_p = best_model.predict(x_te.values)
            te_y_rf2 = pd.DataFrame(te_y_p, index=y_te.index)
            print(set_name, ' fit done')
            # add test performance 
            perf['nmse'].append(mean_squared_error(y_te, te_y_p))
            perf['r2'].append(r2_score(y_te, te_y_p))
            p_corr, _ = pearsonr(y_te.values.flatten(), te_y_p.flatten())
            perf['pr'].append(p_corr)
            perf['mae'].append(mean_absolute_error(y_te, te_y_p))

            # Calculate SHAP values
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(x_tr.values)
            print(set_name, ' shap done')
            #save to disk
            performance = pd.DataFrame(perf)
            # fill output dict
            rf2_dict[set_name]['data'].update({'xtrain': x_tr, 'xtest': x_tr,  'yptrain': tr_y_rf2, 'yptest': te_y_rf2, 'yttrain': y_tr, 'yttest': y_te}) 
            rf2_dict[set_name]['model'].update({'perf': performance,  'shap': shap_values}) 
            rf2_model[set_name]['model'].update({'best_model': best_model,'cv_results': search.cv_results_})
        else:
            print('best_model is none')
            # plt.close('all')
        return best_model
    except Exception as e:
        logging.error(traceback.format_exc())


# %%
### contrasts list:
contrasts_mid= [    
    'Reward-Neutral',
    'Loss-Neutral',
    'LgReward-Neutral',
    'SmallReward-Neutral',
    'LgLoss-Neutral',
    'SmallLoss-Neutral',
    'LgLoss-SmallLoss',
    'LgReward-SmallReward',
    'RewardHit-RewardMiss',
    'LossHit-LossMiss']
contrasts_sst= [    
    'CorrectGo',
    'IncorrectGo',
    'CorrectStop',
    'IncorrectStop',
    'CorrectStop-CorrectGo',
    'IncorrectStop-CorrectGo',
    'Stop-CorrectGo',
    'CorrectStop-IncorrectStop',
    'IncorrectGo-CorrectGo',
    'IncorrectGo-IncorrectStop']
contrasts_wm= [    
    'place',
    'face',
    'emotionface',
    'face-place',
    'PosFace-NeutFace',
    'NegFace-NeutFace',
    'emotionface-NeutFace',
    'twobk',
    'zerobk',
    'twobk-zerobk']
contrasts3_wm= [    
    'twobk',
    'zerobk',
    'twobk-zerobk']


# %%
moda_stack_dict = dict(

                        FC = ['gfc', 'tfc'], 
                        Str2 = ['cort', 'subc', 'surf', 'VolBrain'],
                        WmCntr2 = contrasts3_wm,
                        All8 = ['tfc', 'cort', 'subc', 'surf', 'VolBrain'] + contrasts3_wm,
                        All9 = ['gfc', 'cort', 'subc', 'surf', 'VolBrain'] + contrasts3_wm,
                        All10 = ['conn_wm', 'cort', 'subc', 'surf', 'VolBrain'] + contrasts3_wm,
     
                        )
# moda_stack_dict = dict(
#                         Str = ['cort', 'subc', 'surf', 'rest', 'VolBrain'],
#                         WmCntr = contrasts_wm
#                         )
#, 'part_rest','part_wm','part_mid', 'part_sst', 'tang_rest','tang_wm','tang_mid', 'tang_sst'

# %%

print('started to calculate the Fold #', fold_num, '\n')

targ_list = ['total_','cryst_','fluid_'] #
for targ in targ_list:
    target = joblib.load(main_fold + targ + 'std_targets.joblib')

    smri = joblib.load(enet1_dir + targ + 'smri_enet1_output_std.joblib')
    cntr = joblib.load(enet1_dir + targ + 'cntr_enet1_output_std.joblib')
    conn = joblib.load(enet1_dir + targ + 'conn_enet1_output_std.joblib')
    gtfc = joblib.load(enet1_dir + targ + 'gtfc_enet1_output_std.joblib')   
    # conn2 = joblib.load(enet1_dir + targ + 'con2_enet1_output.joblib')
    # tang = joblib.load(enet1_dir + targ + 'tang_enet1_output.joblib')
    All_enet1 = {**smri, **cntr, **conn, **gtfc} # , **tang, **conn2
    # enet output dict for each target
    rf2_dict = {}
    rf2_model = {}
    for set_name, names in moda_stack_dict.items():
        set_dict = {key: All_enet1[key] for key in All_enet1.keys() if any(substring in key for substring in names)}
        
        # Extract the indices from the DataFrames
        train_ind = set()
        test_ind =  set()
        # for moda in set_dict.values():
        #     train_ind.update(moda['yptrain'].index)
        #     test_ind.update(moda['yptest'].index)
        # Perform a full outer join on the indices
        set_df_tr = None
        set_df_te = None
        for moda_n , moda in set_dict.items():
            moda['data']['yptrain'].columns = [f"{moda_n}" for col in moda['data']['yptrain'].columns]
            moda['data']['yptest'].columns = [f"{moda_n}" for col in moda['data']['yptrain'].columns]
            if set_df_tr is None and set_df_te is None:
                set_df_tr = moda['data']['yptrain']
                set_df_te = moda['data']['yptest']
            else:
                set_df_tr = set_df_tr.join(moda['data']['yptrain'], how='outer', sort=True)
                set_df_te = set_df_te.join(moda['data']['yptest'], how='outer', sort=True)
        # # If there are duplicate columns due to the join, rename them
        # set_df_tr.columns = pd.io.parsers.ParserBase({'names':set_df_tr.columns})._maybe_dedup_names(set_df_tr.columns)
        # set_df_te.columns = pd.io.parsers.ParserBase({'names':set_df_te.columns})._maybe_dedup_names(set_df_te.columns)
        # set_df_tr.to_csv(enet2_dir + targ + set_name + '_df.csv')
        # Reorder columns
        if set_df_tr is not None and set_df_te is not None:
            
            set_df_tr = set_df_tr[sorted(set_df_tr.columns)]
            set_df_te = set_df_te[sorted(set_df_te.columns)]
            
            #replace nan with extremes
            tr_l = set_df_tr.fillna(1000)
            tr_s = set_df_tr.fillna(-1000)
            set_df_tr = tr_l.join(tr_s, lsuffix='_l', rsuffix='_s')
            te_l = set_df_te.fillna(1000)
            te_s = set_df_te.fillna(-1000)
            set_df_te = te_l.join(te_s, lsuffix='_l', rsuffix='_s')
            # get train shared x and y indices
            x_tr = set_df_tr.dropna()
            y_tr = target['train']['std'].dropna()
            common_ind = list(set(y_tr.index).intersection(set(x_tr.index)))
            x_tr = x_tr.loc[common_ind]
            y_tr = y_tr.loc[common_ind]
            # get test shared x and y indices
            x_te = set_df_te.dropna()
            y_te = target['test']['std'].dropna()
            common_ind = list(set(y_te.index).intersection(set(x_te.index)))
            x_te = x_te.loc[common_ind]
            y_te = y_te.loc[common_ind]

            # keys for output dict
            if set_name not in rf2_dict:
                rf2_dict[set_name] = {'model': {}, 'data': {}}
                rf2_model[set_name] = {'model': {}}
            print(targ)
            run_RF2(randfor2_dir, targ)
    # save dict
    joblib.dump(rf2_dict, randfor2_dir + targ + 'rf2-3_output_std.joblib', compress=0)
    joblib.dump(rf2_model, randfor2_dir + targ + 'rf2-3_model_std.joblib', compress=0) 
    # Save rf2_dict using pickle
    with open(randfor2_dir + targ + 'rf2-3_output_std.pkl', 'wb') as f:
        pickle.dump(rf2_dict, f)
    with open(randfor2_dir + targ + 'rf2-3_model_std.pkl', 'wb') as f:
        pickle.dump(rf2_model, f)
   




