# %%
#libraries
import pandas as pd
import numpy as np
import os
import sys
import joblib
import gc
import pickle
print('start')
# %%
targets = ['total_','cryst_','fluid_']#
print('targ')
# %%

root_dir = '/nesi/nobackup/uoo03493/farzane/abcd/'
std_dir = root_dir + 'main_std/'
plot_dir = std_dir + '/prediction_plots/'
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir) 
print('address')

# %%
pls_dict = {}
eNet1_dict = {}
# eNet2_dict = {}

for targ in targets:
    print('targ loop', targ)
    cmid_pls = {}
    cmid_enet = {}
    rf2_dict = {}
    print('emp dict')
    for fold in sorted(os.listdir(std_dir)):
        print('fold loop', fold)
        if 'Fold' in fold and any(char.isdigit() for char in fold) and 'tsp' not in fold:# and int(''.join(filter(str.isdigit, fold))) in [16]:
            main_fold = std_dir + fold
            print('main dir')
            pls_dir = main_fold + '/pls/'
            # pls_dir = main_fold + '/pls/'
            enet1_dir = main_fold + '/enet1/'
            # enet2_dir = main_fold + '/enet2/'
            rf2_dir = main_fold + '/RF2/'
            print('cmid loop')
            if fold not in cmid_pls:
                print('fold loop')
                pls_dict[fold] = {'smri' : joblib.load(pls_dir + targ + 'smri_pls_output.joblib'), 'cntr' : joblib.load(pls_dir + targ + 'cntr_pls_output.joblib'),
                                   'conn' : joblib.load(pls_dir + targ + 'conn_pls_output.joblib')}
                eNet1_dict[fold] = {'smri' : joblib.load(enet1_dir + targ + 'smri_enet1_output.joblib'), 'cntr' : joblib.load(enet1_dir + targ + 'cntr_enet1_output.joblib'),
                                   'conn' : joblib.load(enet1_dir + targ + 'conn_enet1_output.joblib')}      
                # eNet2_dict[fold] = {'stacked' : joblib.load(enet2_dir + targ + 'enet2_output.joblib')}     
                rf2_dict[fold] = {'stacked' : joblib.load(rf2_dir + targ + 'nrf2_output_std.joblib')}    
                #cmid_pls[fold] = {'cntr' : joblib.load(pls_dir + targ + 'mnct_pls_output.joblib')} 
                #cmid_enet[fold] = {'cntr' : joblib.load(enet1_dir + targ + 'mnct_enet1_output_std.joblib')}      
    joblib.dump(pls_dict, plot_dir + targ + 'pls.joblib', compress=0) 
    joblib.dump(eNet1_dict, plot_dir + targ + 'eN1.joblib', compress=0) 
    # joblib.dump(eNet2_dict, plot_dir + targ + 'eN2.joblib', compress=0) 
    joblib.dump(rf2_dict, plot_dir + targ + 'ntrf2.joblib', compress=0) 
    #with open(plot_dir + targ + 'parc_pls.pkl', 'wb') as f:
        #pickle.dump(cmid_pls, f)
    gc.collect()


