# Authors: Romaric Thiboud <romaric.thiboud@inserm.fr>
#          Marine Vernet <marine.vernet@inserm.fr>
#
# License: BSD (3-clause)

import os.path as op
from pathlib import Path
from scipy import io
from scipy import stats
import mne
import numpy as np
import os,sys
import glob
from mne.decoding import LinearModel, get_coef, Scaler, CSP
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import pyriemann 
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import clone

from scipy.signal import detrend
from scipy.signal.windows import tukey
from sklearn.decomposition import PCA

from config import *
from decod_fun import *

# Select the classifier to use
classifier = 'HYDRE'  # Options: 'LogR', 'GBT_PCA', 'LDA_CSP', 'HYDRE'

for n_sub, subject in enumerate(subjects_id) :

    directory = Path(path_analyzed_data, subject, 'epochs_megchannels')
    fname = op.join('%s-epo.fif') % (subject)

    epochs = mne.read_epochs(op.join(directory, fname))
    epochs = epochs.pick(picks='mag')

    if classifier == 'HYDRE' or classifier == 'LDA_CSP':
        epochs = epochs.crop(tmin = -2 , tmax =0)
    else:
        epochs = epochs.crop(tmin = -1.5 , tmax =0)  

    epochs_info=epochs.info
    
    X = epochs.copy().get_data()  # MEG signals: n_epochs, n_meg_channels, n_times    
    
    y = epochs.events[:,2]
    y = np.where(y == 20, 1, np.where(y == 40, 0, y))

    shape_array = X.shape
    # Calculate how many rows need to be removed to make the first dimension divisible by 10
    rows_to_remove = shape_array[0] % 10
    # Remove the last rows from the first dimension
    if rows_to_remove > 0 :
        X = X[:-rows_to_remove, :, :]
        y=y[:-rows_to_remove]
    # Remove the last time point to make the last dimension divisible 6 (at 120Hz)
    X = X[:,:,:-1]     

    # Run the classifiers
    if classifier == 'LogR':
        roc_scores, accuracy_scores, f1_scores, results_df, method = LogR_win50_clf(participant_id=subject, fs=epochs.info['sfreq'], X=X, y=y, folds=10)
    if classifier == 'GBT_PCA':
        roc_scores, accuracy_scores, f1_scores, results_df, method = GBT_default_PCA_win50_clf(participant_id=subject, fs=epochs.info['sfreq'], X=X, y=y, folds=10, n_comp=0.99)
    if classifier == 'LDA_CSP':
        roc_scores, accuracy_scores, f1_scores, results_df, method = LDA_CSP_win200_sl50_clf(participant_id=subject, X=X, y=y, folds=10, fs=epochs.info['sfreq'], 
                                                                                            win_len_ms=200, step_ms=50, time_start=0.5, n_comp=20, cov_est= 'epoch', 
                                                                                            reg='oas', csp_out_type= "average_power")
    if classifier == 'HYDRE':
        roc_scores, accuracy_scores, f1_scores, results_df, method = HYDRE_win50_clf(X, y , participant_id=subject, ensemble_method='voting', pad_duration=0.25, fmin=8, 
                                                                        fmax=30, fsteps=2, fs=epochs.info['sfreq'], folds=10, n_comp_pre_riemann=60, 
                                                                        n_comp_post_riemann=200,  n_comp_temp=0.99, n_comp_tfr=0.95, 
                                                                        C_temporal=1, C_tfr=1, C_riemann=1, C_decision=0.1,  
                                                                        win_len_ms=500, tfr_Riemannian_win_ms=200, 
                                                                        step_ms=50, time_start=0.5, printing=True)

    path_scores = Path(path_output, subject, 'meg', 'scores', method)
    path_scores.mkdir(parents=True, exist_ok=True)
        
    save_roc_scores = '%s_meg_roc_auc_scores_%s.npy' % (subject, method)
    save_accu_scores = '%s_meg_accuracy_scores_%s.npy' % (subject, method)
    save_f1_scores = '%s_meg_f1_scores_%s.npy' % (subject, method)
    save_df_results_classifier = '%s_meg_detailed_results_offline_%s.csv' % (subject, method)
    
    np.save(Path(path_scores, save_roc_scores), roc_scores)
    np.save(Path(path_scores, save_accu_scores), accuracy_scores)
    np.save(Path(path_scores, save_f1_scores), f1_scores)

    results_df.to_csv(Path(path_scores, save_df_results_classifier), index=False)

    print('Done', subject)
