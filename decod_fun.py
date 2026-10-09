# Authors: Romaric Thiboud <romaric.thiboud@inserm.fr>
#          Marine Vernet <marine.vernet@inserm.fr> 
#
# License: BSD (3-clause)

import sys
import mne
import numpy as np
import pandas as pd
from mne.decoding import LinearModel, CSP
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from scipy.signal import detrend
from scipy.signal.windows import tukey
from sklearn.decomposition import PCA

# Riemannian geometry
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

######################## TEST MODEL #########################
def LogR_win50_clf(participant_id, X, y, fs=int, folds=10):
    
    method= 'LogR_win50'
    
    scaler = StandardScaler()
    clf = make_pipeline(scaler, LinearModel(LogisticRegression(max_iter=1000)))
    cv = StratifiedKFold(folds, shuffle=True)  

    roc_scores = list()
    accuracy_scores = list()
    f1_scores = list()

    all_decoding_data = list()  # Dictionary to store decoding data for each fold and time point

    step = 6 # around 50ms
    fold_number = 0  # Initialize fold counter 
       
    for train, test in cv.split(X, y): 
           
        roc_scores_time = list()
        accuracy_scores_time = list()
        f1_scores_time = list()
        
        fold_number += 1  # Increment fold number 
        
        print("\033[91mFold Number: {}\033[00m".format(fold_number))  # Print fold number in red
                 
        for time_train in range(0, X.shape[2], step):
            print(time_train)
            X_train = X[train]
            X_test = X[test] 
            
            X_train= np.squeeze(np.mean(X_train[:,:,time_train:time_train+step], axis=2))
            X_test= np.squeeze(np.mean(X_test[:,:,time_train:time_train+step], axis=2)) 

            clf.fit(X_train, y[train])
            
            pred_label_test = clf.predict(X_test)
            label_proba_test = clf.predict_proba(X_test)
            label_proba_test = label_proba_test[:,1]                       

            roc_score = roc_auc_score(y[test], label_proba_test)
            f1_score_decision = f1_score(y[test], pred_label_test)
            accu_score = accuracy_score(y[test], pred_label_test)
            
            # 4. Save every trial for this specific time point
            y_test = y[test]  # True labels for the test set
            for i, original_trial_idx in enumerate(test):
                all_decoding_data.append({
                    'participant_id': participant_id, # Define this earlier
                    'fold': fold_number,
                    'time': -1500 + int((time_train/fs)*1000),
                    'trial_idx': original_trial_idx,
                    'y_true': y_test[i],
                    'pred': pred_label_test[i],
                    'proba': label_proba_test[i],
                })
                
            roc_scores_time.append(roc_score)
            accuracy_scores_time.append(accu_score)
            f1_scores_time.append(f1_score_decision)

        roc_scores.append(np.array(roc_scores_time))
        accuracy_scores.append(np.array(accuracy_scores_time))
        f1_scores.append(np.array(f1_scores_time))

    # 5. Convert to DataFrame once all folds and time points are done
    results_df = pd.DataFrame(all_decoding_data)

    roc_scores = np.mean(roc_scores, axis=0)
    accuracy_scores = np.mean(accuracy_scores, axis=0)
    f1_scores = np.mean(f1_scores, axis=0)
    
    return roc_scores, accuracy_scores, f1_scores, results_df, method

def LDA_CSP_win200_sl50_clf(participant_id, X, y, folds=10, fs=int, win_len_ms=200, step_ms=50, time_start=0.5, n_comp=20, cov_est= 'epoch', reg='OAS', csp_out_type= "average_power"):
    """
    Tangent Space decoding (TS + Logistic Regression) using PyRiemann,
    with sliding 200ms windows stepping every 50ms.
    """
    method = 'LDA_CSP_%s_win%s_sl%s' % (n_comp, win_len_ms, step_ms)
    
    csp = CSP(n_components= n_comp, reg=reg, log=True, cov_est=cov_est, transform_into= csp_out_type,  norm_trace=False)
    csp_pipeline = make_pipeline(csp, verbose=False)
    scaler = StandardScaler()
    clf = make_pipeline(scaler, LinearDiscriminantAnalysis())
    
    cv = StratifiedKFold(folds, shuffle=True, random_state=42)

    win_len = int((win_len_ms / 1000) * fs)  # Convert ms to samples
    print("Window length (samples):", win_len)
    step = int((step_ms / 1000) * fs)        # Convert ms to samples
    print("Step size (samples):", step)
    
    all_decoding_data = list()  # Dictionary to store decoding data for each fold and time point

    roc_scores = []
    accuracy_scores = []
    f1_scores = []
    
    fold_number = 0  # Initialize fold counter
    
    for train_idx, test_idx in cv.split(X, y):
        
        roc_scores_time = []
        accuracy_scores_time = []
        f1_scores_time = []
        
        start_win = int((time_start*fs))  # Calculate starting window index based on time_start in seconds
        print("Start decod index:", start_win)
        fold_number += 1  # Increment fold number 
        print(f"\033[91mFold Number: {fold_number}\033[00m")


        for time_train in range(start_win, X.shape[2], step):

            print("Window to decode (ms):", -2000 + int((time_train/fs)*1000) ,"-", -2000 + int(((time_train + step)/fs)*1000))
            
            start = time_train - win_len
            end = time_train
            
            X_train_win = X[train_idx, :, start+step : end+step]
            X_test_win  = X[test_idx, :, start+step : end+step]
        
            # ---- 1) CSP  ----
            X_train_transformed = csp_pipeline.fit_transform(X_train_win, y[train_idx]) 
            X_test_transformed = csp_pipeline.transform(X_test_win)

            clf.fit(X_train_transformed, y[train_idx])
            
            pred_label_test = clf.predict(X_test_transformed)
            label_proba_test = clf.predict_proba(X_test_transformed)
            label_proba_test = label_proba_test[:,1]                       

            roc_score = roc_auc_score(y[test_idx], label_proba_test)
            f1_score_decision = f1_score(y[test_idx], pred_label_test)
            accu_score = accuracy_score(y[test_idx], pred_label_test)
            
            # 4. Save every trial for this specific time point
            y_test = y[test_idx]  # True labels for the test set
            for i, original_trial_idx in enumerate(test_idx):
                all_decoding_data.append({
                    'participant_id': participant_id, # Define this earlier
                    'fold': fold_number,
                    'time': -2000 + int((time_train/fs)*1000),
                    'trial_idx': original_trial_idx,
                    'y_true': y_test[i],
                    'pred': pred_label_test[i],
                    'proba': label_proba_test[i],
                })
                
            roc_scores_time.append(roc_score)
            accuracy_scores_time.append(accu_score)
            f1_scores_time.append(f1_score_decision)

        roc_scores.append(np.array(roc_scores_time))
        accuracy_scores.append(np.array(accuracy_scores_time))
        f1_scores.append(np.array(f1_scores_time))

    # 5. Convert to DataFrame once all folds and time points are done
    results_df = pd.DataFrame(all_decoding_data)

    roc_scores = np.mean(roc_scores, axis=0)
    accuracy_scores = np.mean(accuracy_scores, axis=0)
    f1_scores = np.mean(f1_scores, axis=0)
    
    return roc_scores, accuracy_scores, f1_scores, results_df, method

def GBT_default_PCA_win50_clf(participant_id, fs, X, y, folds=10, n_comp=0.99):
    
    method = 'GBT_default_PCA_%s_win50' % (n_comp)
       
    scaler = StandardScaler()
    GBC = GradientBoostingClassifier(random_state=42, verbose=0)
    clf = Pipeline(steps=[("scaler", scaler), ("classifier", GBC)])
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    pca = PCA(n_components=n_comp, random_state=42)

    all_decoding_data = list()  # Dictionary to store decoding data for each fold and time point

    roc_scores = list()
    accuracy_scores = list()
    f1_scores = list()
    
    step = 6 # around 50ms
    fold_number = 0  # Initialize fold counter 
       
    for train, test in cv.split(X, y):   
        
        roc_scores_time = list()
        accuracy_scores_time = list()
        f1_scores_time = list()
        
        fold_number += 1  # Increment fold number 
        
        print("\033[91mFold Number: {}\033[00m".format(fold_number))  # Print fold number in red
               
        for time_train in range(0, X.shape[2], step):

            print("Window to decode (ms):", -1500 + int((time_train/fs)*1000) ,"-", -1500 + int(((time_train + step)/fs)*1000))

            X_train = np.mean(X[train, :, time_train:time_train+step], axis=2)
            X_test = np.mean(X[test, :, time_train:time_train+step], axis=2)

            X_train_tmp = pca.fit_transform(X_train)
            X_test_tmp = pca.transform(X_test)

            X_train_transformed = np.array(X_train_tmp)
            X_test_transformed = np.array(X_test_tmp)
            
            X_train=[]
            X_test=[]
            
            X_train = np.transpose(X_train_transformed, (0, 1))
            X_test = np.transpose(X_test_transformed, (0, 1))

            clf.fit(X_train, y[train])
            
            pred_label_test = clf.predict(X_test)
            label_proba_test = clf.predict_proba(X_test)
            label_proba_test = label_proba_test[:,1]                       

            roc_score = roc_auc_score(y[test], label_proba_test)
            f1_score_decision = f1_score(y[test], pred_label_test)
            accu_score = accuracy_score(y[test], pred_label_test)
            
            # 4. Save every trial for this specific time point
            y_test = y[test]  # True labels for the test set
            for i, original_trial_idx in enumerate(test):
                all_decoding_data.append({
                    'participant_id': participant_id, # Define this earlier
                    'fold': fold_number,
                    'time': -1500 + int((time_train/fs)*1000),
                    'trial_idx': original_trial_idx,
                    'y_true': y_test[i],
                    'pred': pred_label_test[i],
                    'proba': label_proba_test[i],
                })
                
            roc_scores_time.append(roc_score)
            accuracy_scores_time.append(accu_score)
            f1_scores_time.append(f1_score_decision)

        roc_scores.append(np.array(roc_scores_time))
        accuracy_scores.append(np.array(accuracy_scores_time))
        f1_scores.append(np.array(f1_scores_time))

    # 5. Convert to DataFrame once all folds and time points are done
    results_df = pd.DataFrame(all_decoding_data)

    roc_scores = np.mean(roc_scores, axis=0)
    accuracy_scores = np.mean(accuracy_scores, axis=0)
    f1_scores = np.mean(f1_scores, axis=0)
    
    return roc_scores, accuracy_scores, f1_scores, results_df, method

def HYDRE_win50_clf(X, y , participant_id, ensemble_method, pad_duration:float, fmin:int, fmax:int, fsteps:int, fs:int, folds, n_comp_pre_riemann, n_comp_post_riemann, C_temporal, C_tfr, C_riemann, C_decision,  n_comp_temp=0.99, n_comp_tfr=0.95, win_len_ms=500, tfr_Riemannian_win_ms=200, step_ms=50, time_start=0.5, printing=False):   # HYbrid Domain Ensemble Classifier

    method="HYDRE_%s_freq_%s_%s_PCA_T=%s_TFR=%s_R1=%s_R2=%s_C=%s_win50" % (ensemble_method, fmin, fmax, n_comp_temp, n_comp_tfr, n_comp_pre_riemann, n_comp_post_riemann, C_decision) 
    
    scaler_temporal = StandardScaler()
    scaler_tfr = StandardScaler()
    scaler_riemann = StandardScaler()
    
    def fit_pca_on_train(train_data):
        pca_pre_riemann.fit(train_data)
        # Return the actual number of components selected
        return pca_pre_riemann.n_components_

    def apply_pca(data, n_comp_found):
        n_trials, n_channels, n_times = data.shape
        # Reshape: (Trials, Channels, Time) -> (Trials*Time, Channels)
        data_2d = data.transpose(0, 2, 1).reshape(-1, n_channels)
        # Transform using the pre-fitted PCA
        data_pca_2d = pca_pre_riemann.transform(data_2d)
        # Reshape back: (Trials*Time, k) -> (Trials, k, Time)
        return data_pca_2d.reshape(n_trials, n_times, n_comp_found).transpose(0, 2, 1) 

    pca_temporal=PCA(n_components=n_comp_temp, random_state=42)
    pca_tfr=PCA(n_components=n_comp_tfr, random_state=42)    
    pca_pre_riemann = PCA(n_components=n_comp_pre_riemann, whiten=True, random_state=42)
    pca_post_riemann=PCA(n_components=n_comp_post_riemann, whiten=True, random_state=42)

    LogR_temporal = LogisticRegression( max_iter=10000, random_state=42, C=C_temporal)
    LogR_tfr = LogisticRegression( max_iter=10000, random_state=42, C=C_tfr)
    LogR_Riemann= LogisticRegression( max_iter=10000, random_state=42, C=C_riemann)
    LogR_decision= LogisticRegression( max_iter=10000, random_state=42, C=C_decision)
    
    clf_temporal = make_pipeline( pca_temporal, scaler_temporal, LogR_temporal)
    clf_tfr = make_pipeline(pca_tfr, scaler_tfr, LogR_tfr)
    clf_riemann = make_pipeline( pca_post_riemann, scaler_riemann, LogR_Riemann)

    cv = StratifiedKFold(folds, shuffle=True)  

    # multitaper parameters
    freqs = np.arange(fmin, fmax, fsteps)  # Define frequency range from 8 to 30 Hz in steps of 2 Hz
    
    all_decoding_data = list()  # Dictionary to store decoding data for each fold and time point

    roc_scores = list()
    accu_scores = list()
    f1_scores = list()
    
    win_len = int((win_len_ms / 1000) * fs)  # Convert ms to samples
    win_len_tfr_Riemann = int((tfr_Riemannian_win_ms / 1000) * fs)  # Convert ms to samples
    
    pad_size = int(pad_duration * fs)  # Number of samples to pad on each side
    step = int((step_ms / 1000) * fs)        # Convert ms to samples

    if printing:
        print("Window length (samples):", win_len)
        print("Step size (samples):", step)
    
    fold_number = 0  # Initialize fold counter 
       
    for fold_idx, (train, test) in enumerate(cv.split(X, y)):  
        
        roc_scores_time = list()
        accu_scores_time = list()
        f1_scores_time = list()
        
        fold_number += 1  # Increment fold number 
        if printing:
            print("\033[91mFold Number: {}\033[00m".format(fold_number))  # Print fold number in red
        
        start_win = int((time_start*fs))  # Calculate starting window index based on time_start in seconds
        
        for time_train in range(start_win, X.shape[2], step):
            
            if printing:
                print("Window to decode (ms):", -2000 + int((time_train/fs)*1000) ,"-", -2000 + int(((time_train + step)/fs)*1000))
            
            X_train_temporal= np.squeeze(np.mean(X[train,:,time_train:time_train+step], axis=2))
            X_test_temporal= np.squeeze(np.mean(X[test,:,time_train:time_train+step], axis=2))
            
            #############################################################################################
            #***************** Perform Time-Frequency Decomposition *****************
            #############################################################################################
            
            start = time_train - win_len
            end = time_train
            
            # Select the last 500 ms
            X_train = X[train,:,start+step:end+step]
            X_test = X[test,:,start+step:end+step]
            
            # Using scipy.signal.detrend directly on the last axis
            X_train = detrend(X_train, axis=-1, type='linear')
            X_test = detrend(X_test, axis=-1, type='linear')
            
            # Apply a mirror padding to the data before the time window and after the time window
            X_train = np.pad(X_train, ((0, 0), (0, 0), (pad_size, pad_size)), mode='reflect')
            X_test = np.pad(X_test, ((0, 0), (0, 0), (pad_size, pad_size)), mode='reflect')
            
            # Ensures the outer edges of the padded data drop to zero smoothly
            alpha = 2 * pad_size / X_train.shape[-1]
            window = tukey(X_train.shape[-1], alpha=alpha)
            X_train *= window 
            X_test *= window 
            
            # Apply Multitaper transformation
            tfr_train = mne.time_frequency.tfr_array_multitaper(
                    X_train,                # shape: n_trials, n_sensors, n_times
                    sfreq=fs,
                    freqs=freqs,
                    time_bandwidth=2.0,  
                    output='power',
                    n_jobs=-1,
                    verbose=False
                )
            tfr_test = mne.time_frequency.tfr_array_multitaper(
                    X_test,                # shape: n_trials, n_sensors, n_times
                    sfreq=fs,
                    freqs=freqs,
                    time_bandwidth=2.0,  
                    output='power',
                    n_jobs=-1,
                    verbose=False)
            
            X_tfr_train_tmp = tfr_train  # MEG signals: n_epochs, n_meg_channels, n_freq, n_times           
            # Remove the padding after the time-frequency transformation  
            X_tfr_train_tmp = X_tfr_train_tmp[:, :, :, pad_size:-pad_size]
            X_tfr_train_tmp = X_tfr_train_tmp[:, :, :, -win_len_tfr_Riemann:] # Select last 24 time points (200 ms window)
            X_tfr_train_tmp2 = np.mean(X_tfr_train_tmp, axis=3) # Average over time window
            n_epochs, n_meg_channels, n_freq = X_tfr_train_tmp2.shape
            X_train_tfr = X_tfr_train_tmp2.reshape(n_epochs, n_meg_channels * n_freq)
            
            X_tfr_test_tmp = tfr_test  # MEG signals: n_epochs, n_meg_channels, n_freq, n_times            
            # Remove the padding after the time-frequency transformation  
            X_tfr_test_tmp = X_tfr_test_tmp[:, :, :, pad_size:-pad_size]
            X_tfr_test_tmp = X_tfr_test_tmp[:, :, :, -win_len_tfr_Riemann:] # Select last 24 time points (200 ms window)
            X_tfr_test_tmp2 = np.mean(X_tfr_test_tmp, axis=3) # Average over time window
            n_epochs, n_meg_channels, n_freq = X_tfr_test_tmp2.shape
            X_test_tfr = X_tfr_test_tmp2.reshape(n_epochs, n_meg_channels * n_freq)

            #############################################################################################
            #***************** Prepare Riemannian features *****************
            #############################################################################################

            start_riemann = time_train - win_len_tfr_Riemann
            end_riemann = time_train
            
            X_train_riemann = X[train, :, start_riemann+step:end_riemann+step]
            X_test_riemann  = X[test, :, start_riemann+step:end_riemann+step]

            # ---- 1) PCA to reduce dimensionality ----
            X_train_riemann_2d = X_train_riemann.transpose(0,2,1).reshape(-1, X_train_riemann.shape[1])
            
            ncomp_found = fit_pca_on_train(X_train_riemann_2d)

            X_train_riemann_pca = apply_pca(X_train_riemann, ncomp_found)
            X_test_riemann_pca  = apply_pca(X_test_riemann, ncomp_found)                    
        
            # ---- 2) Covariance matrices (OAS) ----
            cov_train = Covariances('oas').fit_transform(X_train_riemann_pca)
            cov_test  = Covariances('oas').transform(X_test_riemann_pca)
            # ---- 3) Tangent space mapping ----
            ts = TangentSpace(metric='logeuclid')
            ts_train = ts.fit_transform(cov_train)
            ts_test  = ts.transform(cov_test)

            print(f"Riemannian features shape: {ts_train.shape}")

            #############################################################################################
            #***************** Train the 4 model, 3 weak learners and 1 strong learner *****************
            #############################################################################################
            
            clf_temporal.fit(X_train_temporal, y[train])
            clf_tfr.fit(X_train_tfr, y[train])
            clf_riemann.fit(ts_train, y[train])

            # Extract the number of components used in PCA for the temporal and tfr modules
            n_components_temporal = clf_temporal.named_steps['pca'].n_components_
            n_components_tfr = clf_tfr.named_steps['pca'].n_components_     
            
            temporal_label_proba = clf_temporal.predict_proba(X_train_temporal)
            temporal_label_proba = temporal_label_proba[:,1] # keep only positive class (binary classification)      
            tfr_label_proba = clf_tfr.predict_proba(X_train_tfr)
            tfr_label_proba = tfr_label_proba[:,1]
            riemann_label_proba = clf_riemann.predict_proba(ts_train)
            riemann_label_proba = riemann_label_proba[:,1]
            
            temporal_label_proba_test = clf_temporal.predict_proba(X_test_temporal)
            temporal_label_proba_test = temporal_label_proba_test[:,1]
            tfr_label_proba_test = clf_tfr.predict_proba(X_test_tfr)
            tfr_label_proba_test = tfr_label_proba_test[:,1]
            riemann_label_proba_test = clf_riemann.predict_proba(ts_test)
            riemann_label_proba_test = riemann_label_proba_test[:,1]
                    
            # Concatenate the output of the 3 weak learner 
            if ensemble_method == "features":
                train_decision = np.concatenate((scaler_temporal.transform(pca_temporal.transform(X_train_temporal)), 
                                                scaler_tfr.transform(pca_tfr.transform(X_train_tfr)), 
                                                scaler_riemann.transform(pca_post_riemann.transform(ts_train)),
                                                tfr_label_proba.reshape(-1, 1), 
                                                temporal_label_proba.reshape(-1, 1), 
                                                riemann_label_proba.reshape(-1, 1)), axis=1) # keep only positive class (binary classification)   
                
                LogR_decision.fit(train_decision,y[train]) 

                # Concatenate the extracted features from the temporal PCA, the time frequency PCA, the output of the 2 weak learner and the prediction of the previous two time windows
                test_decision = np.concatenate((scaler_temporal.transform(pca_temporal.transform(X_test_temporal)), 
                                        scaler_tfr.transform(pca_tfr.transform(X_test_tfr)), 
                                        scaler_riemann.transform(pca_post_riemann.transform(ts_test)),
                                    tfr_label_proba_test.reshape(-1, 1), 
                                    temporal_label_proba_test.reshape(-1, 1), 
                                    riemann_label_proba_test.reshape(-1, 1)), axis=1) # keep only positive class (binary classification) 
            
                decision_label_proba_test = LogR_decision.predict_proba(test_decision)
                decision_label_proba_test = decision_label_proba_test[:,1] # keep only positive class (binary classification)
                # f1 scores and accurary need label pred not proba
                pred_label_test = LogR_decision.predict(test_decision)                
                LogR_decision.fit(train_decision,y[train]) 
                
            if ensemble_method == "outputs":
                train_decision = np.concatenate((tfr_label_proba.reshape(-1, 1), 
                                                temporal_label_proba.reshape(-1, 1), 
                                                riemann_label_proba.reshape(-1, 1)), axis=1) # keep only positive class (binary classification)
                
                LogR_decision.fit(train_decision,y[train]) 
        
                # Concatenate the output of the 3 weak learner 
                test_decision = np.concatenate((tfr_label_proba_test.reshape(-1, 1), 
                                                temporal_label_proba_test.reshape(-1, 1), 
                                                riemann_label_proba_test.reshape(-1, 1)), axis=1) # keep only positive class (binary classification)
                
                decision_label_proba_test = LogR_decision.predict_proba(test_decision)
                decision_label_proba_test = decision_label_proba_test[:,1] # keep only positive class (binary classification)

                # f1 scores and accurary need label pred not proba
                pred_label_test = LogR_decision.predict(test_decision)
                
            if ensemble_method == "voting_proba":
                
                decision_label_proba_test = (temporal_label_proba_test + tfr_label_proba_test + riemann_label_proba_test) / 3
                
                temporal_predict_label = clf_temporal.predict(X_test_temporal)
                tfr_predict_label = clf_tfr.predict(X_test_tfr)
                riemann_predict_label = clf_riemann.predict(ts_test)
                pred_label_test = (decision_label_proba_test > 0.5).astype(int) # If the averaged probability is greater than or equal to 0.5, predict class 1, otherwise predict class 0
                
            if ensemble_method == "voting_pred":

                temporal_predict_label = clf_temporal.predict(X_test_temporal)
                tfr_predict_label = clf_tfr.predict(X_test_tfr)
                riemann_predict_label = clf_riemann.predict(ts_test)   

                decision_label_proba_test = (temporal_label_proba_test + tfr_label_proba_test + riemann_label_proba_test) / 3
                pred_label_test = ((temporal_predict_label.astype(int) + 
                                 tfr_predict_label.astype(int) + 
                                 riemann_predict_label.astype(int)) >= 2).astype(int)

            # 4. Save every trial for this specific time point
            y_test = y[test]  # True labels for the test set
            for i, original_trial_idx in enumerate(test):
                all_decoding_data.append({
                    'participant_id': participant_id, # Define this earlier
                    'fold': fold_idx+1,
                    'time': -2000 + int((time_train/fs)*1000),
                    'trial_idx': original_trial_idx,
                    'y_true': y_test[i],
                    'pred_temporal': temporal_predict_label[i],
                    'pred_tfr': tfr_predict_label[i],
                    'pred_riemann': riemann_predict_label[i],
                    'pred_ensemble': pred_label_test[i],
                    'proba_temporal': temporal_label_proba_test[i],
                    'proba_tfr': tfr_label_proba_test[i],
                    'proba_riemann': riemann_label_proba_test[i],
                    'proba_ensemble': decision_label_proba_test[i],
                    'n_components_temporal': n_components_temporal,
                    'n_components_tfr': n_components_tfr,

                })
                                    
            roc_score = roc_auc_score(y[test], decision_label_proba_test)
            f1_score_decision = f1_score(y[test], pred_label_test)
            accu_score = accuracy_score(y[test], pred_label_test)
            
            roc_scores_time.append(roc_score)
            accu_scores_time.append(accu_score)
            f1_scores_time.append(f1_score_decision) 
        
        if printing:
            print(roc_scores_time)
              
        roc_scores.append(np.array(roc_scores_time))
        accu_scores.append(np.array(accu_scores_time))
        f1_scores.append(np.array(f1_scores_time))

    # 5. Convert to DataFrame once all folds and time points are done
    results_df = pd.DataFrame(all_decoding_data)

    roc_scores = np.mean(roc_scores, axis=0)
    accu_scores = np.mean(accu_scores, axis=0)
    f1_scores = np.mean(f1_scores, axis=0)
    
    return roc_scores, accu_scores, f1_scores, results_df, method
