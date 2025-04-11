# %%
##import libs
import numpy as np
import plotly
from plotly.subplots import make_subplots
import os
import matplotlib.pyplot as plt #to enable plotting within notebook
from nilearn import image as nimg
from nilearn import plotting 
from bids.layout import BIDSLayout
import bids
from matplotlib.pyplot import figure
import mpld3
import pandas as pd
from pathlib import Path   
import nibabel as nb 
import plotly.express as px
from nilearn.datasets import fetch_icbm152_brain_gm_mask
from nilearn.plotting import plot_roi
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
import seaborn as sns
from scipy.stats import norm
from nilearn.glm.contrasts import compute_contrast
from nilearn.plotting import plot_contrast_matrix
from nilearn.reporting import make_glm_report
from nilearn.interfaces.bids import save_glm_to_bids
from ordered_set import OrderedSet
from nilearn.glm.first_level import first_level_from_bids
import pickle
import os.path
import pathlib
import gc
from nilearn import image
from templateflow import api as tflow
from nilearn.maskers import NiftiMasker
import traceback
import logging
from nilearn.glm.first_level import FirstLevelModel
import nilearn as nl
from nilearn.interfaces.fmriprep import load_confounds
import json
from nilearn import surface
import hcp_utils as hcp
from nilearn.glm.contrasts import compute_fixed_effects


# %% [markdown]
# # Run 1stlevel analysis 

# %%
##Set path to the data folder
os.chdir('/media/hcs-sci-psy-narun/ABCC/')
print("The working directory has been changed!")

data_dir = '/media/hcs-sci-psy-narun/ABCC/fmriresults01/'


# %% [markdown]
# ### set major parameters
# 
# #### change task_label accordingly

# %%
## set variable for each run task 

task_label = "SST"
space_label = "MNI152NLin2009cAsym"
derivatives_folder = "derivatives/fmriprep"
run = ['1' , '2']


# %% [markdown]
# ### load mask for plotting 

# %%
## create mask:
# from nilearn import image
# from templateflow import api as tflow
# from nilearn.maskers import NiftiMasker

# mni_gm = tflow.get('MNI152NLin2009cAsym', desc='brain',# label='GM', 
# resolution=2, suffix='T1w', 
# extension='nii.gz')
# gray_matter_binary = image.math_img('i1 > 0.5', i1=image.load_img(mni_gm))
# masker = NiftiMasker(smoothing_fwhm=6).fit(gray_matter_binary)
# masker.generate_report()

# %% [markdown]
# ### load first level objects from file 

# %%
subject_ID_df  = pd.read_csv(data_dir + 'ABCC_SST_subject_list.tsv', sep = '\t')
subject_ID = list(subject_ID_df['0'])


# %% [markdown]
# ## functions

# %% [markdown]
# ### function: extract desired confounds
# #### chooses confound columns & deletes dummy scans 
# #### returns new confound dataframe & list of dummy scans to fix event file

# %%
## function: extract desired confounds
def manage_confounds(original_confs):

       cosine = original_confs.filter(regex='^cosine',axis=1)
       cosine_regs = list(cosine.columns)
       non_steady = original_confs.filter(regex='^non_steady_state',axis=1)
       a_comp_cor = original_confs.filter(regex='^a_comp_cor_',axis=1)
       a_comp_cor = list(a_comp_cor.columns)
       a_comp_cor = a_comp_cor[0:10]
       #a_comp_cor = ['c_comp_cor_00', 'c_comp_cor_01', 'c_comp_cor_02', 'c_comp_cor_03', 'c_comp_cor_04',
       #               'w_comp_cor_00', 'w_comp_cor_01', 'w_comp_cor_02', 'w_comp_cor_03', 'w_comp_cor_04']
       movement_regs = ['rot_x', 'rot_x_derivative1','rot_y','rot_y_derivative1','rot_z','rot_z_derivative1',
                      'trans_x', 'trans_x_derivative1','trans_y', 'trans_y_derivative1','trans_z', 'trans_z_derivative1']

       desired_confs = a_comp_cor + cosine_regs + movement_regs


       if ~set(desired_confs).issubset(original_confs.columns):
              all_confs = original_confs.columns.values.tolist()
              mising_confs = list(set(desired_confs) - set(all_confs))
              final_confounds = list(OrderedSet(desired_confs) - OrderedSet(mising_confs))
       else:
              final_confounds = desired_confs
       valid_size = [453, 445, 442] #based on scanner type
       if len(original_confs) == valid_size[0]:
              dummy_num = 15
       elif len(original_confs) == valid_size[1]:
              dummy_num = 8
       elif len(original_confs) == valid_size[2]:
              dummy_num = 4
       else:
              dummy_num = 0
       confs_final = original_confs.loc[dummy_num:,final_confounds]
       confs_final = confs_final.reset_index(drop=True)
       confs_final.insert(confs_final.shape[1],'linear_trend', range(dummy_num, original_confs.shape[0]))

       return confs_final, dummy_num

# %% [markdown]
# ### function: create & save plots
# #### plots contrasts maps for effect size and z-score

# %%
## function: create & save plots
def manage_plots(unparced_z, unparced_eff , sub_id, contrast_t, task_label, run, output_dir):
     surface_zs = nl.plotting.view_surf(hcp.mesh.inflated,
     hcp.cortex_data(unparced_z), 
     bg_map=hcp.mesh.sulc, cmap='bwr')
     surface_zs.save_as_html(output_dir + '/sub-%s_task-%s_run-%s_contrast-%s_z_parcelated_plot.html' %(sub_id, task_label, run, contrast_t))
     surface_eff = nl.plotting.view_surf(hcp.mesh.inflated,
     hcp.cortex_data(unparced_eff), 
     bg_map=hcp.mesh.sulc, cmap='bwr')
     surface_eff.save_as_html(output_dir + '/sub-%s_task-%s_run-%s_contrast-%s_effect_parcelated_plot.html' %(sub_id, task_label, run, contrast_t))
     plt.close('all')
     gc.collect()

# %% [markdown]
# ### function: prepare events file
# #### delete faulty scans based on fmriprep confound file and fix timings 

# %%
## function: prepare events file:
def prepare_events(ind_events,dummy_scans, t_r):
    n_dummy = dummy_scans
    if n_dummy != 0:
        #print(n_dummy)
        new_zero = n_dummy * t_r
        ind_events['onset'] = ind_events['onset'] - new_zero
        invalid_row_ind = np.asarray(ind_events['onset'] < 0).nonzero()
        #print('invalid rows:' ,invalid_row_ind[0])
        if len(invalid_row_ind[0]) != 0:
            del_row = []
            for i in invalid_row_ind:
                #print(i)
                ii=i[0] 
                new_dur = ind_events.at[ii,'onset']+ind_events.at[ii,'duration']
                if new_dur > 0:
                    ind_events.at[ii,'onset'] = 0 
                    ind_events.at[ii,'duration'] = new_dur
                else:
                    del_row.append(ii)
            ind_events = ind_events.drop(del_row)  
    
    return  ind_events

# %% [markdown]
# ### function: prepare contrast matrix:
# #### defines the desired contrast to be computed later

# %%
# function: prepare contrast matrix:
def prepare_contrasts(design_mat):
   
    #print(event_design_map.shape)
    contrast_matrix = np.eye(design_mat.shape[1])
    contrasts = {
    column: contrast_matrix[i]
    for i, column in enumerate(design_mat.columns)
    }
    contrasts = {
    'CorrectGo': (
    contrasts['CorrectGo'] ),
    'IncorrectGo': (
    contrasts['IncorrectGo']),
    'CorrectStop': (
    contrasts['CorrectStop']),
    'IncorrectStop': (
    contrasts['IncorrectStop']),
    'CorrectStop-CorrectGo': (
    contrasts['CorrectStop']
    - contrasts['CorrectGo'] 
    ),
    'IncorrectStop-CorrectGo': (
    contrasts['IncorrectStop']
    - contrasts['CorrectGo'] 
    ),
    'Stop-CorrectGo': (
    0.5*contrasts['CorrectStop']
    + 0.5*contrasts['IncorrectStop'] 
    - contrasts['CorrectGo']
    ),
    'CorrectStop-IncorrectStop': (
    contrasts['CorrectStop']
    - contrasts['IncorrectStop']
    ),
    'IncorrectGo-CorrectGo': (
    contrasts['IncorrectGo'] 
    - contrasts['CorrectGo'] 
    ),
    'IncorrectGo-IncorrectStop': (
    contrasts['IncorrectGo']
    - contrasts['IncorrectStop'] 
    )}
    return contrasts


# %%
# function: Reshape, Parcellate 
def R_P (stat_map):
    ##parcelate full
    col = np.ones(len(stat_map))
    map_2d=np.stack((stat_map,col),axis=1)
    map_parc_data = np.transpose(map_2d)

    parced_map = hcp.parcellate(map_parc_data, hcp.mmp)
    unparced_map = hcp.unparcellate(parced_map[0], hcp.mmp)

    return parced_map, unparced_map

# %%
# function:  convert to cifti 
def convert_to_cifti (stat_map, axes):
    ##prepare save as cifti
    map = np.reshape(stat_map, (-1, stat_map.shape[0]))
    #save contrasts as cifti full
    scalar_axis_map = nb.cifti2.ScalarAxis(['stat-map']) 
    map_header = nb.Cifti2Header.from_axes([scalar_axis_map, axes[1]])
    map_img = nb.Cifti2Image(map, header = map_header, ) 
                
    return map_img
# %%
# function:  fixed effect
def combine_fixed_effects(contrast_arrays, variance_arrays):
    # Convert to numpy arrays for element-wise operations
    contrast_arrays = np.array(contrast_arrays)
    variance_arrays = np.array(variance_arrays)
    variance_arrays[variance_arrays == 0] = 1e-9
    # Calculate weights as inverse of variances
    weights = 1.0 / variance_arrays

    # Weighted sum of contrasts
    weighted_contrast_sum = np.sum(weights * contrast_arrays, axis=0)

    # Sum of weights
    sum_of_weights = np.sum(weights, axis=0)

    # Combined contrast
    combined_contrast = weighted_contrast_sum / sum_of_weights

    # Combined variance
    combined_variance = 1.0 / sum_of_weights

    return combined_contrast, combined_variance
# %% [markdown]
# ### function: run GLM and extract contrasts
# #### main function 

# %%
## function: run GLM and extract contrasts
def surf_contrast(iter, sub_label, run, task_label, data_dir, output_dir):
    try:
        warnings.filterwarnings(action='ignore')
        complete_run = True
        path = (output_dir+'sub-{}'.format(sub_label))
        #print(path)
        if not os.path.exists(path):
            os.mkdir(path)
        #plt.ioff() 
        json_file = data_dir + 'derivatives/fmriprep/sub-%s/ses-baselineYear1Arm1/func/sub-%s_ses-baselineYear1Arm1_task-%s_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json' %(sub_label,sub_label,task_label)
        with open(json_file, 'r') as f:
            t_r = json.load(f)['RepetitionTime']

        # set up lists for averaging across direction
        z_data = []
        eff_data = []
        v_data = []

        print('start:'+sub_label)

        run_num = 2
        for run_ind in range(run_num):
            fmriprep_bold = (data_dir + 'derivatives/abcd-hcp-pipeline/sub-%s/ses-baselineYear1Arm1/func/sub-%s_ses-baselineYear1Arm1_task-%s_run-%s_bold_timeseries.dtseries.nii' %(sub_label, sub_label, task_label, run[run_ind]))
            eve_dir = (data_dir + 'sst_events/sub-%s_ses-baselineYear1Arm1_task-%s_run-%s_events.tsv' %(sub_label, task_label, run[run_ind]))
            conf_dir = (data_dir + 'derivatives/fmriprep/sub-%s/ses-baselineYear1Arm1/func/sub-%s_ses-baselineYear1Arm1_task-%s_run-%s_desc-confounds_timeseries.tsv' %(sub_label, sub_label, task_label, run[run_ind]))

            events = pd.read_csv(eve_dir , sep='\t')
            confounds = pd.read_csv(conf_dir, sep='\t')


            # call confounds function:
            [confounds_final, dummy_scans] = manage_confounds(confounds)
            if dummy_scans == 0:
                with open('size_log.txt', 'w') as f:
                    f.write('insufficient volumes = %s' %str(confounds_final.shape[0]))
                complete_run = False
                continue

            # calculate frametimes using confounds file n_rows and T_R
            frame_times = (np.arange(confounds_final.shape[0]) * t_r) #+ (t_r/2)

            # call prepare event file function
            ind_events = events.copy()


            # fix events names
            ind_events = ind_events.replace(regex=['0'],value='zero')
            ind_events = ind_events.replace(regex=['2'],value='two')

            # save events file
            # ind_events.to_csv(output_dir + 
            #                 'sub-%s/sub-%s_task-%s_run-%s_events.tsv' %(sub_label, sub_label, task_label, run[run_ind]), sep='\t')
            # confounds_final.to_csv(output_dir + 
            #                 'sub-%s/sub-%s_task-%s_run-%s_confs.tsv' %(sub_label, sub_label, task_label, run[run_ind]), sep='\t')
            # smoothed_img = image.smooth_img(ind_image, 6)
            design_matrix = make_first_level_design_matrix(frame_times = frame_times, 
                                                                events=ind_events, hrf_model='spm', 
                                                                add_regs=confounds_final,
                                                                drift_model= None, #drift_order=2, #high_pass=1/128, 
                                                                # #fir_delays=[0], #add_reg_names=None, #min_onset=-24, #oversampling=50
            )

            #design_matrix.to_csv(data_dir + 'derivatives/nilearn_glm/sub-%s_task-%s_run-%s_design' % (model.subject_label, task_label, direction[dir]))
            del confounds, events

            # load image file and remove dummy scans:
            cifti = nb.load(fmriprep_bold)
            cifti_data = cifti.get_fdata()
            cifti_hdr = cifti.header
            nifti_hdr = cifti.nifti_header
            axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]
            cifti_data = cifti_data[dummy_scans:,:]
            del cifti
            cifti_data = nl.signal.clean(cifti_data, detrend=False, standardize='zscore_sample', confounds=None, standardize_confounds=False,
                                        filter=False, low_pass=None, high_pass=None, t_r=t_r, ensure_finite=False )
            # fit the model 
            labels, estimates = run_glm(cifti_data, design_matrix.values, noise_model='ols')
                
            # design_plot = plotting.plot_design_matrix(design_matrix)
            # design_plot.figure.savefig(path + '/sub-%s_task-%s_run-%s_design.svg' %(sub_label, task_label, run[run_ind]))
            # design_matrix.to_csv(path + '/sub-%s_task-%s_run-%s_design.tsv' %(sub_label, task_label, run[run_ind]), sep='\t')
            
            del cifti_data

            # call contrast matrix function:
            contrasts = prepare_contrasts(design_matrix)
            
            #print("contrast created")
            # compute and generate contrasts (betas and z score)
            for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
                try:
                    contrast_map = compute_contrast(labels, estimates, contrast_val,contrast_type='t')
                    # We present the Z-transform of the t map.
                    zmap = contrast_map.z_score()
                    effect_size = contrast_map.effect_size()[0]
                    variance = contrast_map.effect_variance()
                    
                    z_data.append(zmap)
                    eff_data.append(effect_size)
                    v_data.append(variance)

                    #plot contrast design
                    contrast_plot = plot_contrast_matrix(
                    contrast_val,
                    design_matrix,
                    colorbar=True,
                    )
                    contrast_plot.set_xlabel(contrast_id)
                    contrast_plot.figure.set_figheight(2)
                    contrast_plot.figure.set_tight_layout(True)
                    contrast_plot.figure.savefig(path + '/sub-%s_task-%s_run-%s_contrast-%s_design.svg' %(sub_label, task_label, run[run_ind], contrast_id))
                    
                    parced_z, unparced_z = R_P(zmap)
                    parced_eff, unparced_eff = R_P(zmap)

                    headers = list(hcp.mmp.labels.values())[1:]
                    # pd.DataFrame(parced_z, columns = headers).to_csv(path + '/sub-%s_task-%s_run-%s_contrast-%s_z_parcelations.tsv' %(sub_label, task_label, run[run_ind], contrast_id), sep='\t')
                    # pd.DataFrame(parced_eff, columns = headers).to_csv(path + '/sub-%s_task-%s_run-%s_contrast-%s_effect_parcelations.tsv' %(sub_label, task_label, run[run_ind], contrast_id), sep='\t')

                    # z_img = convert_to_cifti(zmap, axes)
                    # z_img.to_filename(path + '/sub-%s_task-%s_run-%s_contrast-%s_space-fsLR_den-91k_stat-z.dtseries.nii' %(sub_label, task_label, run[run_ind], contrast_id))
                    # eff_img = convert_to_cifti(effect_size, axes)
                    # eff_img.to_filename(path + '/sub-%s_task-%s_run-%s_contrast-%s_space-fsLR_den-91k_stat-effect.dtseries.nii' %(sub_label, task_label, run[run_ind], contrast_id))
                    # unp_z_img = convert_to_cifti(unparced_z, axes)
                    # unp_z_img.to_filename(path + '/sub-%s_task-%s_run-%s_contrast-%s_space-fsLR_den-91k_stat-z_parcellated.dtseries.nii' %(sub_label, task_label, run[run_ind], contrast_id))
                    # unp_eff_img = convert_to_cifti(unparced_eff, axes)
                    # unp_eff_img.to_filename(path + '/sub-%s_task-%s_run-%s_contrast-%s_space-fsLR_den-91k_stat-effect_parcellated.dtseries.nii' %(sub_label, task_label, run[run_ind], contrast_id))
                    
                    # plot contrast maps for each direction
                    #manage_plots(unparced_z, unparced_eff, sub_label, contrast_id, task_label, run[run_ind], path)

                    #print("plots created")

                    plt.close('all')
                except Exception as e:
                    print(f"An error occurred for contrast {contrast_id}: {e}")
                    continue
        del zmap , effect_size, parced_z, unparced_z, parced_eff, unparced_eff, contrast_plot #, z_img, eff_img, unp_z_img, unp_eff_img, 
        plt.close('all')

        if complete_run:
            r2 = int(len(z_data)/2)
            dirM = 'mean'
            for ind , contr_id in enumerate(contrasts.keys()):

                # Compute fixed effects
                mean_z_arr, var_map_z = combine_fixed_effects([z_data[ind], z_data[ind+r2]], [v_data[ind], v_data[ind+r2]])  
                mean_effect_arr, var_map_eff = combine_fixed_effects([eff_data[ind], eff_data[ind+r2]],[v_data[ind], v_data[ind+r2]]) 

                parced_Mz, unparced_Mz = R_P(mean_z_arr)
                parced_Meff, unparced_Meff = R_P(mean_effect_arr)

                pd.DataFrame(parced_Mz, columns = headers).to_csv(path + '/sub-%s_task-%s_run-%s_contrast-%s_z_parcelations.tsv' %(sub_label, task_label, dirM, contr_id), sep='\t')
                pd.DataFrame(parced_Meff, columns = headers).to_csv(path + '/sub-%s_task-%s_run-%s_contrast-%s_effect_parcelations.tsv' %(sub_label, task_label, dirM, contr_id), sep='\t')

                #print("mean created")
                mean_z_img = convert_to_cifti(mean_z_arr, axes)
                mean_z_img.to_filename(path + '/sub-%s_task-%s_run-%s_contrast-%s_space-fsLR_den-91k_stat-z.dtseries.nii' %(sub_label, task_label, dirM, contr_id))
                mean_eff_img = convert_to_cifti(mean_effect_arr, axes)
                mean_eff_img.to_filename(path + '/sub-%s_task-%s_run-%s_contrast-%s_space-fsLR_den-91k_stat-effect.dtseries.nii' %(sub_label, task_label, dirM, contr_id))

                unp_Mz_img = convert_to_cifti(unparced_Mz, axes)
                unp_Mz_img.to_filename(path + '/sub-%s_task-%s_run-%s_contrast-%s_space-fsLR_den-91k_stat-z_parcellated.dtseries.nii' %(sub_label, task_label, dirM, contr_id))
                unp_Meff_img = convert_to_cifti(unparced_Meff, axes)
                unp_Meff_img.to_filename(path + '/sub-%s_task-%s_run-%s_contrast-%s_space-fsLR_den-91k_stat-effect_parcellated.dtseries.nii' %(sub_label, task_label, dirM, contr_id))

                #manage_plots(unparced_Mz, unparced_Meff, sub_label, contr_id, task_label, dirM, path)
                plt.close('all')

        #print("mean plotted")

        print('done:'+sub_label)
        
        gc.collect()
    except Exception as e:
        #error.append(sub_label)
        logging.error(traceback.format_exc())
    # except ValueError:
    #     print(model.subject_label +'raised value error')
    # except TypeError:
    #     print(model.subject_label +'raised type error')
    # except MemoryError:
    #     print(model.subject_label +'raised memory error')    



# %% [markdown]
# ### run in parallel for all subjects

# %%
## run parallel
import warnings
import joblib
from joblib import Parallel, delayed
from joblib import parallel_backend
from joblib import Memory
warnings.filterwarnings(action='ignore')

plt.ioff()
output_dir = "/media/hcs-sci-psy-narun/ABCC/fmriresults01/derivatives/nilearn_glm/surf_SST_std_fixed/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
# model_and_args = zip(models, models_run_imgs, models_events, models_confounds)
with parallel_backend('loky', n_jobs=50):
    Parallel()(delayed(surf_contrast)(iter, subject_label, run, task_label, data_dir, output_dir) for iter, subject_label in enumerate(subject_ID))



