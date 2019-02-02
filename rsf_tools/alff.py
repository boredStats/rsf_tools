# -*- coding: utf-8 -*-
"""
Functions for (fractional) Amplitude of Low Frequency Fluctuations (f/ALFF)

ALFF:
    Yu-Feng, Zang, et al. "Altered baseline brain activity in 
    children with ADHD revealed by resting-state functional MRI." 
    Brain and Development 29.2 (2007): 83-91.
fALFF:
    Zou, Qi-Hong, et al. "An improved approach to detection of amplitude of
    low-frequency fluctuation (ALFF) for resting-state fMRI: fractional ALFF."
    Journal of neuroscience methods 172.1 (2008): 137-141.
    
Created on Fri Feb  1 09:13:05 2019
"""

import numpy as np
from nilearn.masking import compute_gray_matter_mask, apply_mask
from scipy.signal import butter, sosfilt

def bandpass_filter(timeseries, fs, cutoffs=[0.01, 0.08], order=4):
    """
    Apply a bandpass filter to voxel timeseries data
    
    Parameters
    ----------
    timeseries: numpy vector or array
        If timeseries is an array, rows=timepoints, columns=voxels
    
    fs: float
        Sampling rate of the data
    
    cutoffs: list of 2, optional
        Cutoff frequencies for bandpass filter
    
    order: int, optional
        Butterworth filter order
        
    Returns
    -------
    filt: numpy vector or array
        A vector or array with bandpass-filtered data with size of timeseries
    """
    nyq = fs/2
    butter_cuts = [c / nyq for c in cutoffs]
    sos = butter(order, butter_cuts, btype='bandpass', output='sos')
    filt = sosfilt(sos, timeseries)
    return filt

def calc_alff(timeseries):
    """
    Calcuate ALFF for a bunch of voxels
    
    Parameters
    ----------
    timeseries: numpy array
        Array of timeseries data, where rows=timepoints and columns=voxels
    
    Returns
    -------
    alff_values: numpy vector
        Vector of non-normalized ALFF values
    """
    L = timeseries.shape[0]
    N = np.ceil(np.log2(L)).astype(int)
    
    psd = np.fft.fft(timeseries, n=2**N, axis=0)
    psd_sqrt = np.real(np.sqrt(psd))
    alff_values = np.mean(psd_sqrt, axis=0)
    return alff_values

class ALFF():
    def __init__(self):
        self.smoothing = None
        self.gm_mask = None
        self.bandpass = None
        self.verbosity= 0
        
    def extract_gm_signals(self, subj_scan):
        if self.gm_mask is None:
            self.gm_mask = compute_gray_matter_mask(subj_scan,
                                                    smoothing=self.smoothing,
                                                    verbose=self.verbosity)

        gm_signals = apply_mask(subj_scan, self.gm_mask,
                                smoothing=self.smoothing,
                                verbose=self.verbosity)
        return gm_signals
    
    def ALFF(self, subj_scan, fs):
        gm_timeseries = self.extract_gm_signals(self, subj_scan)
        if self.bandpass is not None:
            cleaned_timeseries = bandpass_filter(gm_timeseries, fs,
                                                 self.bandpass)
        else:
            cleaned_timeseries = gm_timeseries
            
        raw_alff = calc_alff(cleaned_timeseries)
        norm_val = np.mean(raw_alff)
        
        alff = raw_alff / norm_val
        return alff

if __name__ == "__main__":
    #Add performance testing
    print('')