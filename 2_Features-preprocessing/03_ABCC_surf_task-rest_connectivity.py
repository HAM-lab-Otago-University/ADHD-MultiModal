# %%
##import libs
import numpy as np
import plotly
from plotly.subplots import make_subplots
import os
import matplotlib.pyplot as plt #to enable plotting within notebook
import matplotlib
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
import cv2

import nibabel as nb
from pathlib import Path              
from pprint import pprint             
import numpy as np                    
from matplotlib import pyplot as plt  
from nilearn import plotting as nlp   
import transforms3d                   
from scipy import ndimage as ndi      
import nibabel.testing                
import hcp_utils as hcp
##import libs for glm and plots
import nilearn
import warnings
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from nilearn.maskers import NiftiLabelsMasker


# %%
##Set path to the data folder
os.chdir('/media/hcs-sci-psy-narun/ABCC/')
print("The working directory has been changed!")


# %%
## set path & get layout object for later use:
data_dir = '/media/hcs-sci-psy-narun/ABCC/fmriresults01/'


# %%
task_label = "rest"
space_label = "MNI152NLin2009cAsym"
derivatives_folder = "derivatives/fmriprep"
run = ['1' , '2' , '3' , '4']
#direction = ['LR', 'RL']
#contrast_type = 'story-math'


# %%
subject_ID_df  = pd.read_csv(data_dir + 'rsfmri_base_r4_qc_list.tsv', sep = '\t')
subject_ID = list(subject_ID_df['0'])
subject_1 = [subject_ID[0]]
subject_10 = subject_ID[0:10]
noisy_sub = ['NDARINVTDCYY0EZ']



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
       fd = ['framewise_displacement', 'std_dvars']
       desired_confs = a_comp_cor + cosine_regs + movement_regs + fd


       if ~set(desired_confs).issubset(original_confs.columns):
              all_confs = original_confs.columns.values.tolist()
              mising_confs = list(set(desired_confs) - set(all_confs))
              final_confounds = list(OrderedSet(desired_confs) - OrderedSet(mising_confs))
       else:
              final_confounds = desired_confs
       
       #get dummy scan rows  
       if ~set(desired_confs).issubset(original_confs.columns):
              all_confs = original_confs.columns.values.tolist()
              mising_confs = list(set(desired_confs) - set(all_confs))
              final_confounds = list(OrderedSet(desired_confs) - OrderedSet(mising_confs))
       else:
              final_confounds = desired_confs
       valid_size = [391, 383, 380]
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
def convert_to_cifti (stat_map, axes, t_r):
    ##prepare save as cifti
    #map = np.reshape(stat_map, (-1, stat_map.shape[0]))
    map = stat_map
    #save contrasts as cifti full
    scalar_axis_map = nb.cifti2.SeriesAxis(start=0, step=t_r, size=map.shape[0])  # Takes a list of names, one per row
    map_header = nb.Cifti2Header.from_axes([scalar_axis_map, axes[1]])
    map_img = nb.Cifti2Image(map, header = map_header, ) 
                
    return map_img


# %%
## function: flag outliers:
def flag_outliers(confounds):
    fd_threshold = 0.5  #from xcp-d 
    std_DVARS_threshold = 1.5  #from nilearn load_conf doc
    outlier_index = []  # just vols with high fd
    censor_index = []  #all vols to be removed including high motion and surroundings
    for ind , vol in enumerate(confounds['framewise_displacement']):
        if vol > fd_threshold or confounds['std_dvars'][ind] > std_DVARS_threshold: 
            outlier_index.append(ind)
            censor_index.extend([ind-2, ind-1, ind, ind+1])
    for item in outlier_index:
        if item+6 in censor_index:
            censor_index.extend(list(range(item+2 , item+6)))
            
    censor_index = [x for x in censor_index if x > -1 and x < confounds.shape[0]]  
    censor_set = set(censor_index)
    censor_index = list(censor_set)
    censor_index.sort()
    data_ind = list(confounds.index)
    censored = list(OrderedSet(data_ind) - OrderedSet(censor_index))
    censored_mask = np.asarray(censored)  # contains vols left after removing high motion and the surroundings
    bi_mask = list(OrderedSet(data_ind) - OrderedSet(outlier_index))
    bi_mask = [False if s in outlier_index else True for s in data_ind]
    bi_mask = np.asarray(bi_mask) 

    return outlier_index, censor_index, censored_mask, bi_mask


# %%
def compute_conn(parced_sig, path, sub_label, task_label, d , cen):
    headers = list(hcp.mmp.labels.values())[1:]
    ##plot nilearn parced correlation matrix
    correlation_measure = ConnectivityMeasure(kind="correlation")
    correlation_matrix = correlation_measure.fit_transform([parced_sig])[0]

    pd.DataFrame(correlation_matrix, columns = headers).to_csv(path +
                                    '/sub-%s_task-%s_run-%s_space-fsLR_atlas-Glasser_desc-%s_measure-pearsoncorrelation_conmat.tsv' %(sub_label, task_label, d, cen), sep='\t')
    # plot connectivity matrix
    conmat_fig = plotting.plot_matrix(
        correlation_matrix,
        figure=(10, 8),
        labels=headers, 
        vmax=1,
        vmin=-1,
        #title="Confounds",
        #reorder=True,
    )
    conmat_fig.figure.savefig(path + '/sub-%s_task-%s_run-%s_space-fsLR_atlas-Glasser_desc-%s_measure-pearsoncorrelation_conmat.png' %(sub_label, task_label, d, cen))
    gc.collect()


# %%
def despiking(signal):
    cen_desp = signal.copy()
    spike_threshold = 3.0
    for i in range(cen_desp.shape[1]):  # loop over columns
        vertex_data = cen_desp[:, i]
        mean_value = np.mean(vertex_data)
        spike_indices = np.where(np.abs(vertex_data - mean_value) > spike_threshold)[0]
        
        # Calculate mean without spike values
        non_spike_indices = np.setdiff1d(np.arange(len(vertex_data)), spike_indices)
        if len(non_spike_indices) > 0:
            replacement_value = np.mean(vertex_data[non_spike_indices])
        else:
            replacement_value = mean_value  # fallback if all are spikes
        
        vertex_data[spike_indices] = replacement_value
        cen_desp[:, i] = vertex_data  # reassign modified column

    return cen_desp


# %%
## function: run GLM and extract contrasts
def surf_conn(iter, sub_label, dir, task_label, data_dir, output_dir):
    try:

        path = (output_dir+'sub-{}'.format(sub_label))
        #print(path)
        if not os.path.exists(path):
            os.mkdir(path)
        #plt.ioff()
        json_file = data_dir + 'derivatives/fmriprep/sub-%s/ses-baselineYear1Arm1/func/sub-%s_ses-baselineYear1Arm1_task-%s_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json' %(sub_label,sub_label,task_label)
        with open(json_file, 'r') as f:
            t_r = json.load(f)['RepetitionTime']

        # set up lists for averaging across direction
        cen_res_list = []
        cenp_res_list = []
        uncen_res_list = []
        censored_vols = []
        outlier_vols = []
        half_vols = []
        valid_ind = []
        complete_run = True
        print('start:'+sub_label)

        dir_num = len(dir)
        for dir_ind in range(dir_num):
            # specify directories
            fmriprep_bold = (data_dir + 'derivatives/abcd-hcp-pipeline/sub-%s/ses-baselineYear1Arm1/func/sub-%s_ses-baselineYear1Arm1_task-%s_run-%s_bold_timeseries.dtseries.nii' %(sub_label, sub_label, task_label, run[dir_ind]))
            conf_dir = (data_dir + 'derivatives/fmriprep/sub-%s/ses-baselineYear1Arm1/func/sub-%s_ses-baselineYear1Arm1_task-%s_run-%s_desc-confounds_timeseries.tsv' %(sub_label, sub_label, task_label, run[dir_ind]))

            #load event and confounds
            confounds = pd.read_csv(conf_dir, sep='\t')

            # call confounds function:
            [confounds_final, dummy_scans] = manage_confounds(confounds)
            if dummy_scans == 0:
                complete_run = False
                censored_vols.append(999)
                outlier_vols.append(999)
                half_vols.append(999)
                continue
            # calculate frametimes using confounds file n_rows and T_R
            full_size = confounds_final.shape[0]
            frame_times = (np.arange(full_size) * t_r) #+ (t_r/2)

            confounds_final.to_csv(output_dir + 
                            'sub-%s/sub-%s_task-%s_run-%s_confs.tsv' %(sub_label, sub_label, task_label, dir[dir_ind]), sep='\t')
            
            # identify and extract outliers
            outliers, censor_ind, censored_mask, bi_mask = flag_outliers(confounds_final)
            
            censored_vols.append(len(censor_ind))
            outlier_vols.append(len(outliers))
            half_vols.append(full_size/2)
            pd.DataFrame(bi_mask).to_csv(path +
                                            '/sub-%s_task-%s_run-%s_outliers.tsv' %(sub_label, task_label, dir[dir_ind]), sep='\t')
            
            confounds_final = confounds_final.drop(['framewise_displacement', 'std_dvars'],axis=1,inplace=False) # drop fd and dvars cols before creating design mat
            scaler = StandardScaler()
            confounds_final_std = pd.DataFrame(scaler.fit_transform(confounds_final), columns = confounds_final.columns)

            # Create design matrix with confounds
            # create design mat
            design_matrix = make_first_level_design_matrix(
                                                            frame_times, #events=ind_events,
                                                            hrf_model=None, drift_model= None,
                                                            add_regs=confounds_final_std)

            # remove all flagged vols/ motion outliers from design mat for censored
            cenlist = []
            desmat_ind = design_matrix.index.tolist()
            for i , ind in enumerate(desmat_ind):
                if i in censor_ind:
                    cenlist.append(ind)
            cen_des_mat = design_matrix.drop(cenlist,axis=0,inplace=False)

            # add outliers to design mat for uncensored
            for n , ind in enumerate(outliers):
                col_name = 'outlier%s' %str(n)
                out_vals = np.zeros(len(confounds_final))
                out_vals[ind] = 1
                design_matrix.insert(design_matrix.shape[1], col_name , out_vals)

            design_plot = plotting.plot_design_matrix(design_matrix)
            design_plot.figure.savefig(path + '/sub-%s_task-%s_run-%s_design.svg' %(sub_label, task_label, dir[dir_ind]))
            design_plot2 = plotting.plot_design_matrix(cen_des_mat)
            design_plot2.figure.savefig(path + '/sub-%s_task-%s_run-%s_censored-design.svg' %(sub_label, task_label, dir[dir_ind]))
            cen_des_mat.to_csv(path + '/sub-%s_task-%s_run-%s_censored-design.tsv' %(sub_label, task_label, dir[dir_ind]), sep='\t')
            design_matrix.to_csv(path + '/sub-%s_task-%s_run-%s_design.tsv' %(sub_label, task_label, dir[dir_ind]), sep='\t')

            del confounds#, events

            # load image file and remove dummy scans:
            cifti = nb.load(fmriprep_bold)
            cifti_data = cifti.get_fdata()
            cifti_hdr = cifti.header
            nifti_hdr = cifti.nifti_header
            axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]
            cifti_data = cifti_data[dummy_scans:,:]     #uncensored
            #standardize:
            std_uncen = nl.signal.clean(cifti_data, detrend=False, standardize='zscore_sample', confounds=None, standardize_confounds=False,
                filter=False, low_pass=None, high_pass=None, t_r=t_r, ensure_finite=False)
            #despike:
            std_desp = despiking(std_uncen)
            #scrub:
            std_desp_cen = np.delete(std_desp, censor_ind, axis=0)   #censored - despiked

            del cifti
            
            # fit the model 
            labels, estimates = run_glm(std_desp_cen, cen_des_mat.values, noise_model='ols')  # to censored signal
            res_sig_cen = estimates[0.0].residuals

            labels2, estimates2 = run_glm(std_desp, design_matrix.values, noise_model='ols')  # to uncensored signal
            res_sig_uncen = estimates2[0.0].residuals

            # labels3, estimates3 = run_glm(cen_desp_cifti_data, cen_des_mat.values, noise_model='ols')  # to uncensored signal
            # res_sig_unt = estimates3[0.0].residuals

            del cifti_data, design_matrix, confounds_final, std_uncen, std_desp, std_desp_cen

            # save residuals
            uncen_res_img = convert_to_cifti(res_sig_uncen, axes, t_r)
            uncen_res_img.to_filename(path + '/sub-%s_task-%s_run-%s_space-fsLR_den-91k_desc-uncensoredDespiked_residuals.dtseries.nii' %(sub_label, task_label, dir[dir_ind]))
            # cen_res_img = convert_to_cifti(res_sig_cen, axes, t_r)
            # cen_res_img.to_filename(path + '/sub-%s_task-%s_run-%s_space-fsLR_den-91k_desc-censoredDespiked_residuals.dtseries.nii' %(sub_label, task_label, dir[dir_ind]))

            # interpolate uncensored residual
            # interplotated_res = res_sig_uncen
            # remained_vol = frame_times[bi_mask]
            # remained_x = interplotated_res[bi_mask, :]
            # cubic_spline_fitter = CubicSpline(remained_vol, remained_x)
            # volumes_interpolated = cubic_spline_fitter(frame_times)
            # interplotated_res[~bi_mask, :] = volumes_interpolated[~bi_mask, :]
            # inter_res_img = convert_to_cifti(interplotated_res, axes, t_r)
            # inter_res_img.to_filename(path + '/sub-%s_task-%s_run-%s_space-fsLR_den-91k_desc-interpolatedDenoised_residuals.dtseries.nii' %(sub_label, task_label, dir[dir_ind]))

            # perform filtering
            # sr = 1.0 / t_r
            #filtered_interplotated_res = nl.signal.butterworth(interplotated_res, sr , low_pass=0.08, high_pass=0.01, order=2)

            # parcellate:
            parced_cen_res = hcp.parcellate(res_sig_cen, hcp.mmp)
            parced_uncen_res = hcp.parcellate(res_sig_uncen, hcp.mmp)
            # parced_interplotated_res = hcp.parcellate(interplotated_res, hcp.mmp)
            ## read glasser header from hcp
            headers = list(hcp.mmp.labels.values())[1:]
            # save parcellation
            # pres_img = convert_to_cifti(parced_res, axes, t_r)
            # pres_img.to_filename(path + '/sub-%s_task-%s_run-%s_space-fsLR_den-91k_desc-uncensoredDenoisedParcelated_residuals.dtseries.nii' %(sub_label, task_label, dir[dir_ind]))
            pd.DataFrame(parced_uncen_res, columns = headers).to_csv(path +
                    '/sub-%s_task-%s_run-%s_space-fsLR_atlas-Glasser_desc-uncensoredDespiked_parcelations.tsv' %(sub_label, task_label, dir[dir_ind]), sep='\t')
            pd.DataFrame(parced_cen_res, columns = headers).to_csv(path +
                                            '/sub-%s_task-%s_run-%s_space-fsLR_atlas-Glasser_desc-censoredDespiked_parcelations.tsv' %(sub_label, task_label, dir[dir_ind]), sep='\t')
            # pd.DataFrame(parced_interplotated_res, columns = headers).to_csv(path +
            #                     '/sub-%s_task-%s_run-%s_space-fsLR_atlas-Glasser_desc-censoredDespikedInterp_parcelations.tsv' %(sub_label, task_label, dir[dir_ind]), sep='\t')
            # manage_plots(res_sig_uncen, res_sig_uncen, sub_label, 'res', task_label, dir[dir_ind], path)

            compute_conn(parced_cen_res, path, sub_label, task_label, dir[dir_ind], "censored")
            compute_conn(parced_uncen_res, path, sub_label, task_label, dir[dir_ind] , "uncensored")
            # compute_conn(parced_interplotated_res, path, sub_label, task_label, dir[dir_ind], "censoredInterp")

            cen_res_list.append(res_sig_cen)
            uncen_res_list.append(res_sig_uncen)
            # cenp_res_list.append(interplotated_res)
            del  parced_cen_res , res_sig_uncen , uncen_res_img , res_sig_cen, parced_uncen_res
            gc.collect()
            plt.close('all')

        plt.close('all')
        dirM = 'all'
        if complete_run:
            ## combine directions and calculate mean 
            full_cen_res = np.concatenate(cen_res_list, axis=0)
            parced_full_cen_res = hcp.parcellate(full_cen_res, hcp.mmp)
            pd.DataFrame(parced_full_cen_res, columns = headers).to_csv(path +
                                                '/sub-%s_task-%s_run-%s_space-fsLR_atlas-Glasser_desc-censored_parcelations.tsv' %(sub_label, task_label, dirM), sep='\t')
            compute_conn(parced_full_cen_res, path, sub_label, task_label, dirM, "censored")
        
            full_uncen_res = np.concatenate(uncen_res_list, axis=0)
            parced_full_uncen_res = hcp.parcellate(full_uncen_res, hcp.mmp)
            pd.DataFrame(parced_full_uncen_res, columns = headers).to_csv(path +
                                                '/sub-%s_task-%s_run-%s_space-fsLR_atlas-Glasser_desc-uncensored_parcelations.tsv' %(sub_label, task_label, dirM), sep='\t')
            ##plot nilearn parced correlation matrix
            compute_conn(parced_full_uncen_res, path, sub_label, task_label, dirM, "uncensored")
            plt.close('all')

            # full_cenp_res = np.concatenate((cenp_res_list[0],cenp_res_list[1]), axis=0)
            # parced_full_cenp_res = hcp.parcellate(full_cenp_res, hcp.mmp)
            # pd.DataFrame(parced_full_cenp_res, columns = headers).to_csv(path +
            #                                     '/sub-%s_task-%s_run-%s_space-fsLR_atlas-Glasser_desc-censoredInterp_parcelations.tsv' %(sub_label, task_label, dirM), sep='\t')
            # compute_conn(parced_full_cenp_res, path, sub_label, task_label, dirM, "censoredInterp")
            plt.close('all')

            censored_vols.append(sum(list(np.array(censored_vols)[valid_ind])))
            outlier_vols.append(sum(list(np.array(outlier_vols)[valid_ind])))
            half_vols.append(sum(list(np.array(half_vols)[valid_ind])))
            
            pd.DataFrame({'round': [dir[0], dir[1], dir[2], dir[3] , dirM ], 'del vols': censored_vols, 'N of vols needed': half_vols, 'high motion vols': outlier_vols}).to_csv(path + 
                                '/sub-%s_task-%s_scrubbing_log.tsv' %(sub_label, task_label), sep='\t')
        else:
            pd.DataFrame({'round': [dir[0], dir[1], dir[2], dir[3]], 'del vols': censored_vols, 'N of vols needed': half_vols, 'high motion vols': outlier_vols}).to_csv(path + 
                    '/sub-%s_task-%s_scrubbing_log.tsv' %(sub_label, task_label), sep='\t')
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




# %%
## run parallel
import warnings
import joblib
from joblib import Parallel, delayed
from joblib import parallel_backend

from joblib import Memory
warnings.filterwarnings(action='ignore')

#plt.ioff()
output_dir = data_dir + "derivatives/nilearn_glm/tconnectivity_rest/no-hrf/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

Parallel(n_jobs=40)(delayed(surf_conn)(iter, subject_label, run, task_label, data_dir, output_dir) for iter, subject_label in enumerate(subject_ID))

