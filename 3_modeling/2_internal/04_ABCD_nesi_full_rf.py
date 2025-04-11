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
randfor_dir = main_fold + '/RF2/'
randfor2_dir = main_fold + '/RF2-2/'

# %%
targ_list = ['total_','cryst_','fluid_'] #
for i, targ in enumerate(targ_list):
    # load rf model :
    rf_model1 = joblib.load(randfor_dir + targ + 'rf2_output_std.joblib')
    rf_model2 = joblib.load(randfor2_dir + targ + 'rf2-2_output_std.joblib')
    rf_model2_3 = joblib.load(randfor2_dir + targ + 'rf2-3_output_std.joblib')
    rf_model2_4 = joblib.load(randfor2_dir + targ + 'rf2-4_output_std.joblib')
    rf_model2_5 = joblib.load(randfor2_dir + targ + 'rf2-5_output_std.joblib')
    rf_model2_6 = joblib.load(randfor2_dir + targ + 'rf2-6_output_std.joblib')
    rf_model2_7 = joblib.load(randfor2_dir + targ + 'rf2-7_output_std.joblib')
    rf_model2_8 = joblib.load(randfor2_dir + targ + 'rf2-8_output_std.joblib')
    rf_model = {**rf_model1, **rf_model2, **rf_model2_3, **rf_model2_4, **rf_model2_5, **rf_model2_6, **rf_model2_7, **rf_model2_8}
    joblib.dump(rf_model, randfor_dir + targ + 'nrf2_output_std.joblib', compress=0)


