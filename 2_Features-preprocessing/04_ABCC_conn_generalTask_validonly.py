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
import glob
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
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline


# %%
#create general fc - ABCC:

wm_list = list(pd.read_csv('/media/hcs-sci-psy-narun/ABCC/fmriresults01/withcalib_list.tsv', sep = '\t')['0'])
sst_list = list(pd.read_csv('/media/hcs-sci-psy-narun/ABCC/fmriresults01/ABCC_SST_subject_list.tsv', sep = '\t')['0'])
mid_list = list(pd.read_csv('/media/hcs-sci-psy-narun/ABCC/fmriresults01/ABCC_MID_subject_list3.tsv', sep = '\t')['0'])
rest_list = list(pd.read_csv('/media/hcs-sci-psy-narun/ABCC/fmriresults01/rsfmri_base_r4_qc_list.tsv', sep = '\t')['0'])
shared_el = list(set(wm_list) & set(sst_list) & set(mid_list) & set(rest_list))
all_el = list (set(wm_list + sst_list + mid_list + rest_list))
print(len(shared_el), len(all_el))

# %%


# %%
qc = pd.read_csv('/media/hcs-sci-psy-narun/abcd-data-release-5.1/core/imaging/mri_y_qc_incl.csv', index_col=0)
new_ind_s = [ind.replace('_', '') for ind in qc.index]
qc.index = new_ind_s

qc_base_ind1 = qc[(qc['eventname'] == 'baseline_year_1_arm_1') & (qc['imgincl_mid_include']== 1)].index
qc_base_ind2 = qc[(qc['eventname'] == 'baseline_year_1_arm_1') & (qc['imgincl_sst_include']== 1)].index
qc_base_ind3 = qc[(qc['eventname'] == 'baseline_year_1_arm_1') & (qc['imgincl_nback_include']== 1)].index
qc_base_ind4 = qc[(qc['eventname'] == 'baseline_year_1_arm_1') & (qc['imgincl_rsfmri_include']== 1)].index

shared_qual = list(set(qc_base_ind1) & set(qc_base_ind2) & set(qc_base_ind3) & set(qc_base_ind4))
len(shared_qual)
qc_base_ind = [qc_base_ind1, qc_base_ind2, qc_base_ind3, qc_base_ind4]

# %%
shared_final = list(set(shared_el) & set(shared_qual))
all_final = list(set(all_el) & set(shared_qual))
print(len(shared_final), len(all_final))

# %%
def compute_conn(parced_sig, path, sub_label, cen):
    headers = list(hcp.mmp.labels.values())[1:]
    ##plot nilearn parced correlation matrix
    correlation_measure = ConnectivityMeasure(kind="correlation")
    correlation_matrix = correlation_measure.fit_transform([parced_sig])[0]

    pd.DataFrame(correlation_matrix, columns = headers).to_csv(path +
                                    '/sub-%s_space-fsLR_atlas-Glasser_desc-%s_measure-pearsoncorrelation_conmat.tsv' %(sub_label, cen), sep='\t')
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
    conmat_fig.figure.savefig(path + '/sub-%s_space-fsLR_atlas-Glasser_desc-%s_measure-pearsoncorrelation_conmat.png' %(sub_label, cen))
    gc.collect()


# %%
def GFC(iter, subject_label, path):
    try:
        rest_key = False
        gfc = []
        tfc = []

        outpath = (path+'sub-{}'.format(subject_label))
        #print(path)
        if not os.path.exists(outpath):
            os.mkdir(outpath)

        wm_file = '/media/hcs-sci-psy-narun/ABCC/fmriresults01/derivatives/nilearn_glm/tconnectivity_WM/fullscrubbed-despiked/sub-%s/sub-%s_task-nback_run-all_space-fsLR_atlas-Glasser_desc-uncensored_parcelations.tsv' %(subject_label, subject_label)
        sst_file = '/media/hcs-sci-psy-narun/ABCC/fmriresults01/derivatives/nilearn_glm/tconnectivity_SST/fullscrubbed-despiked/sub-%s/sub-%s_task-SST_run-all_space-fsLR_atlas-Glasser_desc-uncensored_parcelations.tsv' %(subject_label, subject_label)
        mid_file = '/media/hcs-sci-psy-narun/ABCC/fmriresults01/derivatives/nilearn_glm/tconnectivity_MID/fullscrubbed-despiked_onset/sub-%s/sub-%s_task-MID_run-all_space-fsLR_atlas-Glasser_desc-uncensored_parcelations.tsv' %(subject_label, subject_label)
        rest_file = '/media/hcs-sci-psy-narun/ABCC/fmriresults01/derivatives/nilearn_glm/tconnectivity_rest/no-hrf/sub-%s/sub-%s_task-rest_run-all_space-fsLR_atlas-Glasser_desc-uncensored_parcelations.tsv' %(subject_label, subject_label)

        for i , task in enumerate([mid_file, sst_file, wm_file , rest_file]):
            directory_path = task.split('sub-%s' % subject_label)[0] + 'sub-%s/' % subject_label
            log_file = glob.glob(os.path.join(directory_path, '*scrubbing_log*'))
            print(directory_path)
            if os.path.isfile(task) and subject_label in qc_base_ind[i] and log_file:
                log_data = pd.read_csv(log_file[0], sep='\t')
                if np.sum(log_data['del vols'][:-1]) < np.sum(log_data['N of vols needed'][:-1]):
                    table = pd.read_csv(task, sep='\t', index_col=0)
                    ar = np.array(table)
                    gfc.append(ar)
                    if i == 3:
                        rest_key = True
                    else:
                        tfc.append(ar)

        if len(gfc) > 1 and rest_key:
            gfc_tseries = np.vstack(gfc)
            compute_conn(gfc_tseries, outpath, subject_label, "general")
        if len(tfc) > 1:
            tfc_tseries = np.vstack(tfc)
            compute_conn(tfc_tseries, outpath, subject_label, "task")
        gc.collect()
    except Exception as e:
        #error.append(sub_label)
        logging.error(traceback.format_exc())



# %%
## run parallel
import warnings
import joblib
from joblib import Parallel, delayed
from joblib import parallel_backend

from joblib import Memory
warnings.filterwarnings(action='ignore')

output_path = '/media/hcs-sci-psy-narun/ABCC/fmriresults01/derivatives/nilearn_glm/FC-valid/'
if not os.path.exists(output_path):
    os.makedirs(output_path)


Parallel(n_jobs=40)(delayed(GFC)(iter, subject_label, output_path) for iter, subject_label in enumerate(all_el))

