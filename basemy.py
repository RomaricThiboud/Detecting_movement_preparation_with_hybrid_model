
# Authors: Fayed Rassoulou <rasfay@stanford.edu>
#          Marine Vernet <marine.vernet@inserm.fr>
#
# License: BSD (3-clause)

import os
import os.path as op
import matplotlib.pyplot as plt
import numpy as np

def datatodec(which_channels, epochs):
    if which_channels is 'meg':
        data_to_dec = epochs._data
    elif which_channels is 'eyes':
        data_to_dec = epochs._data[:, epochs.ch_names.index('UADC009-2104'):epochs.ch_names.index('UADC010-2104')+1, :]  # noqa
    elif which_channels is 'pupil':
        data_to_dec = epochs._data[:, None, epochs.ch_names.index('UADC013-2104'), :]  # noqa
    elif which_channels is 'trigger':
        data_to_dec = epochs._data[:, None, epochs.ch_names.index('UPPT001'), :]  # noqa
    elif which_channels is 'all':
        data_to_dec = epochs._data
    return data_to_dec

def whattodecod(somecode, y):
    yTD = y.copy()
    yTD[:] = 1
    yTD[np.where(np.in1d(y, somecode))] = 0
    return yTD

def smooth_to_plot(array_of_decod):

    array_of_decod2 = array_of_decod.copy()
    for i in range(len(array_of_decod)-4):
        array_of_decod2[i+2] = (array_of_decod2[i] +
                                array_of_decod2[i+1] +
                                array_of_decod2[i+2] +
                                array_of_decod2[i+3] +
                                array_of_decod2[i+4])/5
    return array_of_decod2

def plotchanceandlimit(horizontal_extent, horizontal_borders, vertical_extent, vertical_borders):  # noqa
    for hor_bor in horizontal_borders:
        plt.plot(horizontal_extent, [hor_bor, hor_bor], color='black', linestyle='dashed', alpha=0.5)  # noqa
    for ver_bor in vertical_borders:
        plt.plot([ver_bor, ver_bor], vertical_extent, color='black', linestyle='dashed', alpha=0.5)  # noqa

def transfo_sign_analog(signal, targetValueMin, targetValueMax, targetStep1, targetStep2, halfDist):  # noqa

    signal2 = np.copy(signal)
    signal_high = np.where([item > (targetValueMax-targetStep1) for item in signal])  # 37.5  # noqa
    signal_low = np.where([item < (targetValueMin+targetStep1) for item in signal]) # 0.25  # noqa
    signal2[signal_high] = targetValueMax  # 37.75
    signal2[signal_low] = targetValueMin  # 0
    signal3 = np.copy(signal2)
    signal_high = np.where([(item < targetValueMax and item > (targetValueMax-targetStep2)) for item in signal2])  # noqa
    signal_low = np.where([(item < (targetValueMin+targetStep2) and item > targetValueMin) for item in signal2])  # noqa
    for item in signal_high[0]:
        if signal2[item+halfDist] > (targetValueMax-targetStep2):
            signal3[item] = targetValueMax
    for item in signal_low[0]:
        if signal2[item+halfDist] < (targetValueMin+targetStep2):
            signal3[item] = targetValueMin
    signal4 = np.copy(signal3)
    signal_inter = np.where([(item < targetValueMax and item > targetValueMin) for item in signal3[:-halfDist]])  # noqa
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
    plt.hist((events_tri[F, 0]-events_ima[F, 0])/acqfreq, bins=40, label='Computer', fc=(1, 0, 0, 0.5))      # distribution of random changes of iamge  
    plt.legend()
    plt.xlabel('time_interval (between image onset and event, similar to RT) (s)')
    plt.ylabel('number of trials')
    plt.title(run_name + ' Distribution of events ('+ when + ' deleting trials)')
    fname = op.join(figuresfolder, '%s_eventbefore.jpg' %(run_name))
    if saving_fig: plt.savefig(fname, dpi=300)
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
