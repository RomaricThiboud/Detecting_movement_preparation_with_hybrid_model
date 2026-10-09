import numpy as np
import csv
import mne
import os.path as op
import os
from mne import Epochs
from autoreject import AutoReject
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from pathlib import Path

from config import *
from basemy import *

all_chan = False

testing = True

if testing :

    subjects_id = [ 's0993']

for nub_subject, subject in enumerate(subjects_id) :
        
    if all_chan : 
        directory = op.join(path_analyzed_data, subject, 'epochs_allchannels')        
    else :
        directory = op.join(path_analyzed_data, subject, 'epochs_megchannels')
        
    # Load epochs from subject
    fname = op.join('%s-epo.fif') % (subject)
    epochs_all = mne.read_epochs(op.join(directory, fname))
    epochs = epochs_all.copy()
    bad_channels = []
    
    if testing :
        epochs.plot()
        
    if subject == 's0986' :
        bad_channels = [ ]
       
    # orignal epochs targets (before ICA and AR)
    orig_epochs = epochs.copy()
    epochs = epochs.drop_channels(bad_channels)
        
    # STEP A: compute ICA  ############################################################
    ica = ICA( max_iter="auto", method='fastica', n_components=100, verbose=True, random_state=97) 
    ica.fit(epochs)
    # Explore ICA --> differents steps to make the final decision on components exclusion
    # ICA step 0:
    explained_var_ratio = ica.get_explained_variance_ratio(epochs)
    for channel_type, ratio in explained_var_ratio.items():
        print(f"Fraction of {channel_type} variance explained by all components: "f"{ratio}")
        
    # can also explore one component specifically
    explained_var_ratio_comp = ica.get_explained_variance_ratio(epochs, components=[0], ch_type="mag")
    ratio_percent = round(100* explained_var_ratio_comp["mag"])
    print(f"Fraction of variance in EEG signal explained by the component number 0: "f"{ratio_percent}")
    explained_var_ratio_comp2 = ica.get_explained_variance_ratio(epochs, components=[2], ch_type="mag")
    ratio_percent2 = round(100* explained_var_ratio_comp2["mag"])
    print(f"Fraction of variance in EEG signal explained by the component number 3: "f"{ratio_percent2}")
    
    # ICA step 1: 
    # # Try to find the ECG components in ICA
    ecg_indices = list()   
    try:
        ecg_indices, ecg_scores = ica.find_bads_ecg(epochs, method='ctps')
    except ValueError as e:
        print("Error:", e)
        print("Continuing without ECG artifact correction.")
        # Perform other steps of analysis without ECG artifact correction
        pass
    
    ica.exclude = ecg_indices
    ica.exclude = list(set(ica.exclude))
    print(ica.exclude)
    
    # ICA step 2:
    # # look at components time series to visually investigates blinks or saccades
    # Plot ICA topographies components
    if testing :
        ica.plot_components()  
        ica.plot_sources(epochs, show_scrollbars=True, block= True)   
        print(ica.exclude)
    
    # ICA step 3:
    # show overlay of the original signal against the reconstructed signal with the artifactual excluded  
    if testing :    
        mne.viz.plot_ica_overlay(ica, epochs["nothing"].average(), exclude= ica.exclude) 
        mne.viz.plot_ica_overlay(ica, epochs["button"].average(), exclude= ica.exclude) 

    # ICA step 5:
    # Finally we can choose which components to remove from the signal    
    if subject == 's0986' :
        ica.exclude = [1, 9, 22]  
    if subject == 's0987' :
        ica.exclude = [1, 5, 28 ] 
    if subject == 's0989' :
        ica.exclude = [0, 1, 63, 66, 76 ]   
    if subject == 's0990' :
        ica.exclude = [0, 5 ]
    if subject == 's0991' :
        ica.exclude = [0, 2, 21 ]
    if subject == 's0992' :
        ica.exclude = [0, 3, 5 ]
    if subject == 's0993' :
        ica.exclude = [0, 1, 8, 89 ]
    if subject == 's0994' :
        ica.exclude = [0, 1, 8, ]
    if subject == 's0995' :
        ica.exclude = [0, 7, 15 ]
    if subject == 's0996' :
        ica.exclude = [1, 2, 12 ]
    if subject == 's1116' :
        ica.exclude = [2, 6 ]
    if subject == 's1117' :
        ica.exclude = [1, 7 ]        
    if subject == 's1120' :
        ica.exclude = [8, 33, 27, 0, 25, ] 
    if subject == 's1121' :
        ica.exclude = [6, 0, 14, ]  
    if subject == 's1124' :
        ica.exclude = [0, 1, 5, 9 ]   
    if subject == 's1127' :
        ica.exclude = [0, 13, 17, 37, 66, ]      
    if subject == 's1129' :
        ica.exclude = [13, 17, 26, ]                         
    if subject == 's1130' :
        ica.exclude = [0, 1, 3, 2, 15, 51, ]
    
    exclude = ica.exclude
    
    # ICA step 6:
    # apply ICA solution on the epochs
    epochs_test = ica.apply(epochs, exclude=ica.exclude) # epochs is modify here   

    # Save the epochs after ICA correction
    directory = Path(path_analyzed_data / subject / 'epochs_megchannels_ICA')
    # Create parent directories if they don't exist
    directory.mkdir(parents=True, exist_ok=True)
    fname2 = directory / f'{subject}_ICA-epo.fif'
    epochs_test.save(fname2, overwrite=True)      

    # STEP B: Run Autoreject (on epochs with ICA applied)  ##################################################
    ar_step2 = AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                        n_jobs=-1, verbose=True)
    ar_step2.fit(epochs_test)  
    epochs_ar_step2, reject_log_ar_step2 = ar_step2.transform(epochs_test, return_log=True) 
  
    print(f'Number of epochs originally: {len(orig_epochs)}, '
    f'after autoreject: {len(epochs_ar_step2)}') 
      
    if testing :   
        reject_log_ar_step2.plot('horizontal')
        reject_log_ar_step2.plot_epochs(orig_epochs, scalings=dict(eeg=5e-5))  
        
        if len(orig_epochs[reject_log_ar_step2.bad_epochs]) != 0 :     
            evoked_bad_ar_step2 = orig_epochs[reject_log_ar_step2.bad_epochs].average()
            plt.figure()
            plt.plot(evoked_bad_ar_step2.times, evoked_bad_ar_step2.data.T * 1e6, 'r', zorder=-1)
            epochs_ar_step2.average().plot(axes=plt.gca())
            plt.show()
        # Other visu of the preprocess steps
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        # Plot each of the four averages in a grid
        orig_epochs["button"].average().plot(axes=axs[0, 0], show=False)
        orig_epochs["nothing"].average().plot(axes=axs[0, 1], show=False)
        epochs_ar_step2["button"].average().plot(axes=axs[1, 0], show=False)
        epochs_ar_step2["nothing"].average().plot(axes=axs[1, 1], show=False)
        # Add legend to each subplot
        axs[0, 0].legend(["orig_epochs button"])
        axs[0, 1].legend(["orig_epochs nothing"])
        axs[1, 0].legend(["epochs_ar_step2 button"])
        axs[1, 1].legend(["epochs_ar_step2 nothing"])
        # Adjust layout
        plt.tight_layout()
        # Save the plot as an image file
        plt.savefig("C:/Users/Romar/Documents/PhD_Agency/DATA/data_analyzed/figures/Preprocessing/%s_before_after_ar_by_condition_meg.svg" % (subject))

    
    # STEP C: Save the cleaned epochs
    if all_chan :
        fname = 'epochs_allchannels'
    else :
        fname = 'epochs_megchannels'
        
    directory = op.join(path_analyzed_data, subject, fname)
    fname2 = op.join(directory,'%s_clean-epo.fif'
                % (subject))
    epochs_ar_step2.save(fname2, overwrite=True)
    print(fname2)
    
    
    
    # Create a file to keep a track of %good vs bad epochs, how many segments were interpolated, 
    # how many epochs at the end of the preproc, how many components rejected from the ica... 
    # ica components
    ica_compo_rejected=len(exclude)
    # epochs at the end of the prepro
    nbre_epochs_after_prepro=len(epochs_ar_step2)
    # bad vs good epochs after the prepro
    nbre_bad_epochs=len(epochs)-len(epochs_ar_step2)
    percentage_bad_epochs=1-(len(epochs_ar_step2)/len(epochs))
    # proportion of segments bad and interpolated / bad only or good 
    nbr_seg_interpolated=0
    nbr_seg_bad=0
    nbr_seg_good=0

    for line in reject_log_ar_step2.labels:
        for i in line:
            if i == 2:
                nbr_seg_interpolated+=1
            if i == 1: 
                nbr_seg_bad+=1
            if i ==0:
                nbr_seg_good+=1
                
    segments_tot = len(epochs)*64
    percentage_seg_bad=nbr_seg_bad/segments_tot
    percentage_seg_interpo=nbr_seg_interpolated/segments_tot
    percentage_seg_good=nbr_seg_good/segments_tot

    # save into one object  
    # Define the path to the csv file
     
    if all_chan:
        csv_file_path = 'all_chan_metrics_summary.csv'
    else:
        csv_file_path = 'meg_chan_metrics_summary.csv'

    metrics_filename = os.path.join(path_analyzed_data, csv_file_path)

    # Create or open the csv file in append mode
    with open(metrics_filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Check if the file is empty
        is_empty = os.path.getsize(metrics_filename) == 0

        if is_empty:
            csvwriter.writerow(['Metric', 'Value', 'Subject'])

        # Writing the metrics
        csvwriter.writerow(['ICA Components Rejected', ica_compo_rejected, subject])
        csvwriter.writerow(['Epochs After Preprocessing', nbre_epochs_after_prepro, subject])
        csvwriter.writerow(['Bad Epochs', nbre_bad_epochs, subject])
        csvwriter.writerow(['Percentage Bad Epochs', percentage_bad_epochs, subject])
        csvwriter.writerow(['Bad Segments', nbr_seg_bad, subject])
        csvwriter.writerow(['Interpolated Segments', nbr_seg_interpolated, subject])
        csvwriter.writerow(['Good Segments', nbr_seg_good, subject])
        csvwriter.writerow(['Percentage Bad Segments', percentage_seg_bad, subject])
        csvwriter.writerow(['Percentage Interpolated Segments', percentage_seg_interpo, subject])
        csvwriter.writerow(['Percentage Good Segments', percentage_seg_good, subject])
        
