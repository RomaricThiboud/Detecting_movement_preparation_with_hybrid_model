import numpy as np
import mne
from mne.io import read_raw_ctf
import os.path as op
from pathlib import Path
import os
from mne import Epochs
import matplotlib.pyplot as plt


# **** path ************************************************************
computer = 'MEG_data' 

if computer == 'MEG_data':
    path_data = 'N:/impact/Personnel/Marine/DATA/Exp1_Agency'
    path_analyzed_data = 'C:/Users/Romar/Documents/PhD_Agency/DATA/data_analyzed'
    path_figures = 'C:/Users/Romar/Documents/PhD_Agency/DATA/data_analyzed/figures/plot_decoding'
if computer == 'EEG_data' :
    path_data = 'N:/impact/Personnel/Marine/DATA/Exp1_bis_EEG'
    path_analyzed_data = 'C:/Users/Romar/Documents/PhD_Agency/DATA/data_analyzed_eeg'

# **** subjects ************************************************************

if computer == 'MEG_data' :

    subjects_id = ['s0986', 's0987', 's0989', 's0990', 's0991', 's0992', 's0993', 's0994',
                    's0995', 's0996', 's1116', 's1117', 's1120', 's1121', 's1124', 's1127', 's1129', 's1130']
    
    subjects_id = ['s0986']

# **** parameters *****************************************
which_channels = 'meg'
epo_type = 'other' #'riemannian'  

epoching = False
plotting = True
saving_fig = False

# **** functions *******************************************************

def transfo_sign_analog(signal, targetValueMin, targetValueMax, targetStep1, targetStep2, halfDist): 
    signal2 = np.copy(signal)
    signal_high = np.where([item > (targetValueMax-targetStep1) for item in signal])  
    signal_low = np.where([item < (targetValueMin+targetStep1) for item in signal]) 
    signal2[signal_high] = targetValueMax  
    signal2[signal_low] = targetValueMin  
    signal3 = np.copy(signal2)
    signal_high = np.where([(item < targetValueMax and item > (targetValueMax-targetStep2)) for item in signal2]) 
    signal_low = np.where([(item < (targetValueMin+targetStep2) and item > targetValueMin) for item in signal2]) 
    for item in signal_high[0]:
        if signal2[item+halfDist] > (targetValueMax-targetStep2):
            signal3[item] = targetValueMax
    for item in signal_low[0]:
        if signal2[item+halfDist] < (targetValueMin+targetStep2):
            signal3[item] = targetValueMin
    signal4 = np.copy(signal3)
    signal_inter = np.where([(item < targetValueMax and item > targetValueMin) for item in signal3[:-halfDist]])  
    for item in signal_inter[0]:
        if signal3[item+halfDist] > (targetValueMax-targetValueMin)/2:
            signal4[item] = targetValueMax
        else:
            signal4[item] = targetValueMin
    signal5 = np.copy(signal4)
    signal5[np.where(signal5 == targetValueMin)] = 0
    return signal5

def detect_emg(signal, threshold1, threshold2, length1, length2):
    emg3 = np.abs(signal)
    emg4 = np.zeros(signal.shape)
    for item in np.where(emg3 > threshold1)[0]:
        liminf = np.max([0, item - length1])
        limsup = np.min([item + length1, emg3.shape[0]])
        if np.where(emg3[liminf:limsup]>threshold2)[0].shape[0]>1:
            emg4[liminf:limsup] = threshold1
    emg5 = emg4
    for item in np.where(emg4 == threshold1)[0]:
        if emg4[item+length2]==threshold1:
            emg5[item:item+length2] =threshold1
            
    return emg5

def plot_eyetracker(raw_eyeA, raw_eyeB, raw_eyeC, raw_eyeD, run, file, acqfreq, plotting, saving_fig):
    timems = np.arange(0, raw_eyeA.shape[0]/acqfreq, 1/acqfreq)
    plt.plot(timems, raw_eyeB, alpha=0.7, label='X')
    plt.plot(timems, raw_eyeA, alpha=0.7, label='Y')
    plt.plot(timems, raw_eyeC, alpha=0.7, label='Pupil')
    plt.plot(timems, raw_eyeD, alpha=0.7, label='last')
    plt.xlabel('Time (s)')
    plt.ylabel('Arbitrary value')
    plt.title(run + ' Eye Tracker')
    plt.legend(loc='upper right')
    fname = op.join(file, '%s_eye.jpg' % run)
    if saving_fig: plt.savefig(fname, dpi=300)
    if plotting: plt.show()

def plot_photodiode(photodiode, photodiode2, trigger, run, plotting):
    plt.plot(photodiode)
    plt.plot(photodiode2)
    plt.plot(trigger)
    plt.title(run + ' Photodiode and Trigger')
    plt.legend(('Original', 'Transfored', 'Trigger'), loc='upper left')
    if plotting: plt.show()

def plot_detect_emg(emg, emg2, trigger, run, plotting):
    plt.plot(emg)
    plt.plot(emg2)
    plt.plot(trigger)
    plt.xlabel('time points')
    plt.ylabel('Arbitrary value')
    plt.title(run + ' EMG')
    plt.legend(('EMG', 'EMG detect', 'Event'), loc='upper left')
    if plotting: plt.show()

def plot_photo_emg(photodiode2, emg2, trigger, run_name, plotting):
    plt.plot(photodiode2, label='photodiode2')
    plt.plot(trigger, label='trigger')
    plt.plot(emg2, label='emg2')
    plt.title(run_name + ' Photodiode, EMG and Trigger')
    plt.legend(loc='upper right')
    if plotting: plt.show()

def plot_event(events_tri, events_ima, E, F, run_name, figuresfolder, acqfreq, plotting, saving_fig, when):
    plt.hist((events_tri[E, 0]-events_ima[E, 0])/acqfreq, bins=40, label='Human', alpha = 0.5)      # distribution of reaction time  
    plt.hist((events_tri[F, 0]-events_ima[F, 0])/acqfreq, bins=40, label='Computer', fc=(1, 0, 0, 0.5))      # distribution of random changes of image  
    plt.legend()
    plt.xlabel('time_interval (between image onset and event, similar to RT) (s)')
    plt.ylabel('number of trials')
    plt.title(run_name + ' Distribution of events ('+ when + ' deleting trials)')
    fname = op.join(figuresfolder, '%s_eventbefore.jpg' %(run_name))
    if saving_fig: plt.savefig(fname, dpi=300, )
    if plotting: plt.show()

def plot_event_diff(A, B, run_name, acqfreq, plotting, title):
    plt.hist((A-B)/acqfreq, bins=40)
    plt.xlabel('time interval (s)')
    plt.ylabel('number of trials')
    plt.title(run_name + title)
    if plotting: plt.show()

def plot_emg_before_button(G, acqfreq, run_name, plotting):
    plt.hist(G/acqfreq, bins=40)
    plt.xlabel('time (s)')
    plt.ylabel('number of trials')
    plt.title(run_name + ' Event minus Detection_EMG')
    if plotting: plt.show()

def plot_event_allsubj(events_allsubj_array, figuresfolder, acqfreq, plotting=True, saving_fig=False):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gamma
    import os.path as op

    # Indices for human and computer
    human_index = np.where(events_allsubj_array[:, 2] == 20)[0]
    computer_index = np.where(events_allsubj_array[:, 2] == 40)[0]

    # Reaction times in seconds
    rt_human = np.asarray(events_allsubj_array[human_index, 0], dtype=float) / acqfreq
    rt_computer = np.asarray(events_allsubj_array[computer_index, 0], dtype=float) / acqfreq

    # Fit gamma distributions
    shape_human, loc_human, scale_human = gamma.fit(rt_human, floc=0)
    shape_computer, loc_computer, scale_computer = gamma.fit(rt_computer, floc=0)

    # Plot histograms and get counts for scaling
    counts_human, bins_human, _ = plt.hist(rt_human, bins=40, alpha=0.5, label='Human', color='blue')
    counts_computer, bins_computer, _ = plt.hist(rt_computer, bins=40, alpha=0.5, label='Computer', color='red')

    # Bin width for scaling PDF to histogram
    bin_width_human = bins_human[1] - bins_human[0]
    bin_width_computer = bins_computer[1] - bins_computer[0]

    # X-axis for plotting gamma curves
    x = np.linspace(0, max(rt_human.max(), rt_computer.max()), 500)

    # Scale PDF to match histogram counts
    pdf_human = gamma.pdf(x, a=shape_human, loc=loc_human, scale=scale_human) * len(rt_human) * bin_width_human
    pdf_computer = gamma.pdf(x, a=shape_computer, loc=loc_computer, scale=scale_computer) * len(rt_computer) * bin_width_computer

    # Plot gamma curves
    plt.plot(x, pdf_human, 'b-', lw=2, label='Gamma fit Human')
    plt.plot(x, pdf_computer, 'r-', lw=2, label='Gamma fit Computer')

    # Labels, legend, title
    plt.xlabel('Reaction time (s)')
    plt.ylabel('Number of trials')
    plt.title('Distribution of events across all participants')
    plt.legend()

    # Save figure if requested
    fname = op.join(figuresfolder, 'RT_all_subj_project1.svg')
    if saving_fig: 
        plt.savefig(fname, dpi=800, format='svg')
    if plotting: 
        plt.show()

# **** main code ******************************************************

figuresfolder = op.join(path_analyzed_data, 'figures')
if not op.exists(figuresfolder): os.mkdir(figuresfolder)

# **** loading files ****
subj_nb = 0

list_event_allsubj = list()

for subject in subjects_id:

    subj_nb += 1

    # is there a quality photodiode? emg channel?
    quality_emg = 0 # for now let's forget about EMG
    quality_photodiode = 1

    # open the good folder depending on the type of data
    if subject == 's1124' or subject == 's1127':
        folder_name = 'MEGEEG'
    else :
        folder_name = 'MEG'
    
    # select the run to be opened
    fname7 = op.join(path_data, subject, 's1', folder_name)
    files = os.listdir(fname7)
    runs = list()
    runs.extend(f for f in files if ('.ds' in f))
    runs.sort()
    if subject == 's0995':
        runs = runs[0:-1]
    
    if len(runs) == 7:
        runs = runs[1:]

    # Create list to save behavior info across runs
    list_events = list()
    
    for run_number, this_run in enumerate(runs):
        run_to_load = op.join(fname7, this_run)
        raw = read_raw_ctf(run_to_load, preload=True)
        run_name = this_run[:-3]

        # acquisition frequency
        acqfreq = raw.info['sfreq']
        
        print(f"Acquisition frequency for run {run_name}: {acqfreq}")

        # **** reading the triggering channel ****
        trigger_ch_number = raw.ch_names.index('UPPT002')
        trigger = raw.get_data()[trigger_ch_number]

        # **** reading the eye movements channels ****
        raw_eyeA = raw.get_data()[raw.ch_names.index('UADC010-2800')]
        raw_eyeB = raw.get_data()[raw.ch_names.index('UADC011-2800')]
        raw_eyeC = raw.get_data()[raw.ch_names.index('UADC012-2800')]
        raw_eyeD = raw.get_data()[raw.ch_names.index('UADC016-2800')]
        raw_eyes = np.array([raw_eyeA, raw_eyeB, raw_eyeC, raw_eyeD])
        plot_eyetracker(raw_eyeA, raw_eyeB, raw_eyeC, raw_eyeD, run_name, figuresfolder, acqfreq, plotting, saving_fig)
        
        # **** reading the photodiode ****
        if quality_photodiode:
            photodiode_ch_number = raw.ch_names.index('UADC001-2800')
            photodiode = raw.get_data()[photodiode_ch_number]
            photodiode = photodiode*10.0
            
            # remove the fluctuations at the beginning of the first file
            if this_run == '0986-RJ_agency_20210318_01.ds':
                photodiode[:20000] = 0
            elif this_run == '0987-AEG_agency_20210318_01.ds':
                photodiode[:10000] = 0
            elif this_run == '0989MY_agency_20210415_01.ds':
                photodiode[:6000] = 0
            elif this_run == '0990TZ_agency_20210415_01.ds':
                photodiode[:10000] = 0
            elif this_run == '0991MA_agency_20210415_01.ds':
                photodiode[:6000] = 0
            elif this_run == '0992ES_agency_20210429_01.ds':
                photodiode[:10000] = 0
            elif this_run == '0993DAA_agency_20210429_01.ds':
                photodiode[:15000] = 0
            elif this_run == '0994CC_agency_20210429_01.ds':
                photodiode[:5000] = 0
            # for this participant the scale of the photodiode was different
            if subject == 's0989' or subject == 's0990' or subject == 's0991' :
                photodiode2 = transfo_sign_analog(photodiode, 0.5, 16, 0.05, 0.2, 1)
                photodiode2[np.where(photodiode2 == 16)] = 1.3
                if this_run == '0989MY_agency_20210415_01.ds':
                    photodiode2[75550:75650] = 1.3
                    photodiode2[76200:77414] = 1.3
                
            # Data of the participants were recorded with a filter (0.05-30 Hz) --> need to modified the signal of the photodiode 
            elif subject == 's1116' or subject == 's1117' or subject == 's1120' or subject == 's1121' or subject == 's1124' or subject == 's1129' or subject == 's1130' : 
                photodiode2 = transfo_sign_analog(abs(photodiode), 0.5, 1.3, 0.05, 0.2, 1) 
                non_zero_index = np.where(photodiode2 != 0)[0]
                photodiode2[0 : non_zero_index[0]] = 1.3
                photodiode2[non_zero_index[0] : non_zero_index[0]+50] = 0
                photodiode2= 1.3 - photodiode2    
                
            # for this participant the scale of the photodiode was different    
            elif subject == 's1127' :
                photodiode = photodiode*100
                photodiode2 = transfo_sign_analog(abs(photodiode), 0.5, 1.3, 0.05, 0.2, 1) 
                non_zero_index = np.where(photodiode2 != 0)[0]
                photodiode2[0 : non_zero_index[0]] = 1.3
                photodiode2[non_zero_index[0] : non_zero_index[0]+50] = 0
                photodiode2= 1.3 - photodiode2
                                
            else:
                photodiode2 = transfo_sign_analog(photodiode, 0.5, 1.3, 0.05, 0.2, 1)
                if this_run == '0992ES_agency_20210429_01.ds':
                    photodiode2[104900:105100] = 1.3
                    
            plot_photodiode(photodiode, photodiode2, trigger, run_name, plotting)

        # **** feeding back the transformed data ****
        data = raw.get_data()
        if quality_photodiode:
            data[photodiode_ch_number] = photodiode2
        raw._data = data

        # **** detecting the events from the triggering and photodiode channels ****
       
        events_tri = mne.find_events(raw, stim_channel="UPPT002",
                                     consecutive=True,
                                     shortest_event=1)
        
        
        # NB: having a 1 instead of zero allows to detect event towards this value as well
        trigger_ch_number = raw.ch_names.index('UPPT002')
        trigger = raw.get_data()[trigger_ch_number]
        trigger[trigger==0] = 1
        data = raw.get_data()
        data[trigger_ch_number] = trigger
        raw._data = data
        
        events_tri_all = mne.find_events(raw, stim_channel="UPPT002",
                            consecutive=True,
                            shortest_event=1)
    
        
        if quality_photodiode:
            events_ima = mne.find_events(raw, stim_channel="UADC001-2800",
                                         consecutive=True,
                                         shortest_event=1)
            photodiode3 = 1.3 - photodiode2
            data[photodiode_ch_number] = photodiode3
            raw._data = data
            events_bla = mne.find_events(raw, stim_channel="UADC001-2800",
                                         consecutive=True,
                                         shortest_event=1)
            
        # Delete trigger associated to start and end of file 
        events_tri_to_del_id = list()
        for parcours_eve_tri_ix in range(events_tri.shape[0]):
            if events_tri[parcours_eve_tri_ix,2] == 252 or events_tri[parcours_eve_tri_ix,2] == 253 or events_tri[parcours_eve_tri_ix,2] == 4:
                events_tri_to_del_id.append(parcours_eve_tri_ix)
        events_tri = events_tri[np.delete(range(0, events_tri.shape[0]), events_tri_to_del_id)] 
        
        events_tri_to_del_id = list()
        for parcours_eve_tri_ix in range(events_tri_all.shape[0]):
            if events_tri_all[parcours_eve_tri_ix,2] == 252 or events_tri_all[parcours_eve_tri_ix,2] == 253 or events_tri_all[parcours_eve_tri_ix,1] == 252 or events_tri_all[parcours_eve_tri_ix,1] == 253 or events_tri_all[parcours_eve_tri_ix,2] == 4:
                events_tri_to_del_id.append(parcours_eve_tri_ix)
        events_tri_all = events_tri_all[np.delete(range(0, events_tri_all.shape[0]), events_tri_to_del_id)] 

        # for this file, the recordings started late. We miss the onset of at least one image. We need to delete the first event
        if this_run == '0986-RJ_agency_20210318_01.ds':
            events_tri = events_tri[np.delete(range(0, events_tri.shape[0]), 0)] 
            if quality_photodiode:
                events_bla = events_bla[np.delete(range(0, events_bla.shape[0]), 0)] 
                events_ima = events_ima[np.delete(range(0, events_ima.shape[0]), 1)] 

        # for this file, the recordings started late. We miss the onset of at least one image. We need to delete the first event
        if this_run == '0986-RJ_agency_20210318_04.ds':
            events_tri = events_tri[np.delete(range(0, events_tri.shape[0]), 0)] # noqa
            if quality_photodiode:
                events_bla = events_bla[np.delete(range(0, events_bla.shape[0]), 0)] # noqa

        # for this file, glinch of the photodiode
        if this_run == '0989MY_agency_20210415_01.ds':
            events_tri = events_tri[np.delete(range(0, events_tri.shape[0]), 51)] 
            if quality_photodiode:
                events_bla = events_bla[np.delete(range(0, events_bla.shape[0]), 51)] 
                events_ima = events_ima[np.delete(range(0, events_ima.shape[0]), 52)] 
       
        #**** plot the distribution of events ****
        E = np.where(events_tri[:, 2] == 20)[0]  # button
        F = np.where(events_tri[:, 2] == 40)[0]  # nothing
        if quality_photodiode  and plotting:
            plot_event(events_tri, events_ima, E, F, run_name, figuresfolder, acqfreq, plotting, saving_fig, 'before')
            plot_event_diff(events_bla[:, 0], events_tri[:, 0], run_name, acqfreq, plotting, ' time between event and image offset') # 7 values between 20 ms and 30 ms  # noqa
            plot_event_diff(events_ima[1:, 0], events_bla[0:-1, 0], run_name, acqfreq, plotting, ' time of blank') # 22 or 23 ms of blanck for all trials, one trial with 32 ms, except between the pause  # noqa
            

            delay_offset_duration_ms = ((events_bla[:, 0] - events_tri[:, 0])/acqfreq)*1000
            filtered_delay_offset_duration_ms = delay_offset_duration_ms[ delay_offset_duration_ms < 500]
            mean_delay_offset_duration = np.mean(filtered_delay_offset_duration_ms)
            std_delay_offset_duration = np.std(filtered_delay_offset_duration_ms)
            print(f'Averaged delay between trigger and image offset: {mean_delay_offset_duration}, std: {std_delay_offset_duration}')

            blank_duration_ms = ((events_ima[1:, 0] - events_bla[0:-1, 0])/acqfreq)*1000
            filtered_blank_duration_ms = blank_duration_ms[ blank_duration_ms < 500]
            mean_blank_duration = np.mean(filtered_blank_duration_ms)
            std_blank_duration = np.std(filtered_blank_duration_ms)
            print(f'Averaged Blank duration: {mean_blank_duration}, std: {std_blank_duration}')

        # Delete triggers too close to each other (the second one) and the associate images/blanks 
        events_tri_to_del_id = list()
        events_tri_all_to_del_id = list()
        for parcours_eve_tri_ix in range(events_tri.shape[0]-1):
            if abs(events_tri[parcours_eve_tri_ix, 0]-events_tri[parcours_eve_tri_ix+1, 0]) < 0.58 * raw.info['sfreq']:  
                events_tri_to_del_id.append(parcours_eve_tri_ix + 1) 
                
                index_to_remove = np.where(events_tri[parcours_eve_tri_ix+1, 0] == events_tri_all[:, 0])[0][0]
                events_tri_all_to_del_id.append(index_to_remove)
                events_tri_all_to_del_id.append(index_to_remove-1)

        events_tri_all = events_tri_all[np.delete(range(0, events_tri_all.shape[0]), events_tri_all_to_del_id)] 
        events_tri = events_tri[np.delete(range(0, events_tri.shape[0]), events_tri_to_del_id)]         
        
        # here we remove the blank immediately following the trigger (same index as the trigger)
        # for calculating the blank durations, we should also remove the image immediately following (index of the trigger + 1)
        # but for calculating the reaction times, we should keep the onset of this second image and remove the previous image (same index of the trigger)
        if quality_photodiode:
            len_ima_orig = events_ima.shape[0]
            events_ima2 = events_ima
            events_bla = events_bla[np.delete(range(0, events_bla.shape[0]), events_tri_to_del_id)] 
            events_ima = events_ima[np.delete(range(0, events_ima.shape[0]), events_tri_to_del_id)] # for calculating the reaction time 
            cond = 0
            if events_tri_to_del_id[-1] + 1 >= len_ima_orig: # if the last element to remove is the very last triggers
                cond = 1
                events_tri_to_del_id.pop() # in that case there is no image following. Below, you should go up to the last blank (and not the one before the last)
                
            events_ima2 = events_ima2[np.delete(range(0, events_ima2.shape[0]), [x+1 for x in events_tri_to_del_id])] # for calculating blank duration # noqa
            
        # # **** plot (again) the distribution of events ****
        E = np.where(events_tri[:, 2] == 20)[0]  # button
        F = np.where(events_tri[:, 2] == 40)[0]  # nothing
        if quality_photodiode and plotting:
            plot_event(events_tri, events_ima, E, F, run_name, figuresfolder, acqfreq, plotting, saving_fig, 'after')
            if cond == 1:
                plot_event_diff(events_ima2[1:, 0], events_bla[:, 0], run_name, acqfreq, plotting, ' time of blank')
            else:
                plot_event_diff(events_ima2[1:, 0], events_bla[0:-1, 0], run_name, acqfreq, plotting, ' time of blank')
            if plotting: plt.show()

        # Select the start of emg immediately preceding a button press
        if quality_emg:
            G = list()
            for parcours_eve_tri_ix in range(E.shape[0]):
                lastposemg = np.where((events_tri[E[parcours_eve_tri_ix]]-events_emg[:])[:,0]>0)[0][-1]
                G.append((events_tri[E[parcours_eve_tri_ix]]-events_emg[lastposemg])[0])
            G=np.array(G)
            plot_emg_before_button(G, acqfreq, run_name, plotting)

        
        if subject == 's1116' or subject == 's1117' or subject == 's1120' or subject == 's1121' or subject == 's1124' or subject == 's1127' or subject == 's1129' or subject == 's1130':
                
            trig_offset = events_tri_all[(events_tri_all[:, 2] == 1) & ((events_tri_all[:, 1] == 40) | (events_tri_all[:, 1] == 20))] # trigger offset
            trig_onset = events_tri_all[(events_tri_all[:, 1] == 1) & ((events_tri_all[:, 2] == 40) | (events_tri_all[:, 2] == 20))] # trigger onset
            if trig_offset.shape[0] < trig_onset.shape[0]:
                trig_onset = trig_onset[1:,:] # to be sure to have the same number of onsets and offsets
            RT= abs(trig_offset[:,0] - trig_onset[:,0]) 
            col = np.zeros_like(RT)
            
            col4 = np.full_like(col, subject, dtype=object)
            behavior_array = np.column_stack((RT, col, trig_onset[:,2], col4))

        else:
            # Store behavior info
            col_human = np.abs(events_tri[E, 0]-events_ima[E, 0])
            col_computer = np.abs(events_tri[F, 0]-events_ima[F, 0])
            # Second column: zeros (already in arrays, but we can define it explicitly)
            col2_human = np.zeros_like(col_human)
            col2_computer = np.zeros_like(col_computer)
            # Third column: last column of tmp
            col3_human = events_tri[E, -1]
            col3_computer = events_tri[F, -1]
            
            # add participant ID in the fourth column
            col4_human = np.full_like(col_human, subject, dtype=object)
            col4_computer = np.full_like(col_computer, subject, dtype=object)
            
            # Stack into a single array (column-wise)
            tmp1 = np.column_stack((col_human, col2_human, col3_human, col4_human))
            tmp2 = np.column_stack((col_computer, col2_computer, col3_computer, col4_computer))
            behavior_array = np.vstack((tmp1, tmp2))
                    
        list_events.append(behavior_array)
        
        # preparing the events for epoching  
        events_meg = events_tri
        event_id = {'button': 20, 'nothing': 40}

        if epoching:

            # **** epoching ****
            if epo_type == 'riemannian':
                
                # **** filtering ****
                raw.filter(0.05, 30)
                 
                # This is a try for Riemannian decoding
                epochs = Epochs(raw, events_meg, event_id=event_id,
                        tmin=-2, tmax=0.5,
                        preload=True,
                        baseline=None)
                epochs.resample(400)  # downsampling to 400 Hz for Riemannian decoding
                
            else:
                
                # **** filtering ****
                raw.filter(0.05, 30)
                # These are the epochs used for all the analysis
                epochs = Epochs(raw, events_meg, event_id=event_id,
                                tmin=-2, tmax=1,
                                preload=True,
                                baseline=None, decim=5)
            
            # **** saving with everything ****
            
            directory = Path(path_analyzed_data) / subject / 'epochs_allchannels'
            # Create parent directories if they don't exist
            directory.mkdir(parents=True, exist_ok=True)
            fname1 = directory / f'{run_name}_down_sampling_400Hz-epo.fif'
            epochs.save(fname1, overwrite=True)

            # **** saving with just the MEG channels ****
            epochs.pick_types(meg=True)
            directory = Path(path_analyzed_data) / subject / 'epochs_megchannels'
            # Create parent directories if they don't exist
            directory.mkdir(parents=True, exist_ok=True)
            fname2 = directory / f'{run_name}_down_sampling_400Hz-epo.fif'
            epochs.save(fname2, overwrite=True)      
            

    participant_events = np.vstack(list_events)
    print(participant_events.shape)
    list_event_allsubj.append(participant_events)

import pandas as pd

dfs = []  # list to store each participant's DataFrame

for participant_events in list_event_allsubj:
    rt_values = participant_events[:, 0] / acqfreq  # Reaction times
    condition_codes = participant_events[:, 2]      # Condition codes
    participant_ids = participant_events[:, 3]      # Participant IDs

    # Map condition codes to labels
    condition_labels = np.where(condition_codes == 20, 'human', 'computer')

    df_participant = pd.DataFrame({
        "rt": rt_values,
        "condition": condition_labels,
        "condition_num": condition_codes,
        "participant": participant_ids
    })

    dfs.append(df_participant)

    # Concatenate all participants into one DataFrame
    df = pd.concat(dfs, ignore_index=True)

csv_filename = op.join(path_analyzed_data, 'RT_allsubj_project1.csv')
df.to_csv(csv_filename, index=False)

events_allsubj_array = np.vstack(list_event_allsubj)
np.save(op.join(path_analyzed_data, 'RT_allsubj_project1.npy'), events_allsubj_array)

# **** Concatenate the run together ****
if epoching:
    if 'epochs' in locals():
        del epochs

    for subj_nb, subject in enumerate(subjects_id): 
        
            fname3 = op.join(path_data, subject, 's1', 'MEG')
            if subject == 's1124' or subject == 's1127':
                fname3 = op.join(path_data, subject, 's1', 'MEGEEG')
            files = os.listdir(fname3)
            runs = list()
            runs.extend(f for f in files if ('.ds' in f))
            runs.sort()
            if subject == 's0995':
                runs = runs[0:-1]
            if subject == 's0996':
                runs = runs[2:]
            # if subject == 's1124':
            #     runs = runs[0:3,5:7]
            if len(runs) == 7:
                runs = runs[1:]
                
            # ******** loading and concatenating the epochs *********
            for directory in [op.join(path_analyzed_data, subject, 'epochs_allchannels'),
                            op.join(path_analyzed_data, subject, 'epochs_megchannels')]:

                for run_number, this_run in enumerate(runs):
                    run_name = this_run[:-3]
                    fname4 = op.join(directory,'%s_down_sampling_400Hz-epo.fif'% (run_name))

                    if 'epochs' not in locals():
                        epochs = mne.read_epochs(fname4)
                    else:
                        epochs2 = mne.read_epochs(fname4)
                        epochs2.info['dev_head_t'] = epochs.info['dev_head_t']
                        epochs = mne.concatenate_epochs([epochs, epochs2])

                # **** save epochs ****
                if 'epochs' in locals():
                    fname5 = op.join(directory,'%s_down_sampling_400Hz-epo.fif'
                                    % (subject))
                    epochs.save(fname5, overwrite=True)
                    print(fname5)
                    del epochs

plot_event_allsubj(events_allsubj_array, figuresfolder, acqfreq, plotting=True, saving_fig=True)


# Perform testing between the 2 conditions
import scipy.stats as stats
human_index = np.where(events_allsubj_array[:, 2] == 20)[0]
computer_index = np.where(events_allsubj_array[:, 2] == 40)[0]

rt_human = events_allsubj_array[human_index, 0]/acqfreq
rt_computer = events_allsubj_array[computer_index, 0]/acqfreq
rt_human = np.asarray(rt_human, dtype=float)
rt_computer = np.asarray(rt_computer, dtype=float)

u_stat, p_value = stats.mannwhitneyu(rt_human, rt_computer, alternative="two-sided")
print(f"Mann-Whitney U test: U-statistic = {u_stat}, p-value = {p_value}")

from scipy.stats import gamma
shape_human, loc_human, scale_human = gamma.fit(rt_human, floc=0)
shape_computer, loc_computer, scale_computer = gamma.fit(rt_computer, floc=0)
print(f"Human RT - shape: {shape_human}, loc: {loc_human}, scale: {scale_human}")   
print(f"Computer RT - shape: {shape_computer}, loc: {loc_computer}, scale: {scale_computer}")

checking = False
if checking :          
    # Look at the different runs for this participant to identify abberant channels
    subject='s1124'
    fname = op.join(path_analyzed_data, subject, 'epochs_megchannels')
    files = os.listdir(fname)
    runs_epochs = list()
    runs_epochs.extend(f for f in files if ('agency' in f))
    runs_epochs.sort()

    for numb_epo, run_epoch in enumerate(runs_epochs) :
            epochs = mne.read_epochs(op.join(fname, run_epoch))
            evoked=epochs.average()
            fig = evoked.plot()
            
            
            
            