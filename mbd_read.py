"""Loads & sets MBD-Mnist(NOT visual) dataset"""

### EMOTIV EPOC HAS 14 SENSORS (10-20 ARRANGED) sampled at 128hz

import mne
import numpy as np
import pandas as pd

def read_data(data_df, n_epochs):
    
    """Args: data_df"""
    
    n_channels = 14 # channels
    n_times = 128 # Hz
    
    data_list = []
    skip_list = []

    for epoch in range(n_epochs):

        print("Loading epoch for epoch %s" % epoch)

        epoch_slice = data_df["data"].iloc[:n_channels] # get 14 channels * n data points
        epoch_slice = epoch_slice.str.split('\s*,\s*', expand=True) # change to df values
        epoch_slice = epoch_slice.to_numpy() # convert to 2d numpy array
        epoch_slice = epoch_slice[np.newaxis, :, :]
        
        print(epoch_slice.shape)
        
        skip = 0
        
        if epoch_slice.shape[2] < 200:
            skip = epoch
            print(skip)
        else:
            epoch_slice = epoch_slice[:, :,:200]
            print(epoch_slice.shape)
            
        print(skip)
            
        data_list.append(epoch_slice)
        data_df = data_df[14:]
        
    return data_list, skip_list

def read_metadata(data_df, n_epochs):
    
    """Args: data_df"""
    
    metadata_list = []
    
    for epoch in range(n_epochs):
        
        event_id = data_df["code"].iloc[0]
        metadata_list.append(event_id)
        
        data_df = data_df[14:]
        
    metadata_df = pd.DataFrame(metadata_list, columns = ["event_id"])
        
    return metadata_df

def epoch_data(data_list):
    
    """Args: data_list (list of np.arrays corresponding to epochs),
    metadata_array (array of events)"""
    
    # Convert List to np.array
    data_arr = np.asarray(data_list)
    data_arr = np.squeeze(data_arr)
    
    # Setup MNE Epochs
    ch_names = ["AF3", "F7", "F3", "FC5", 
                  "T7", "P7", "O1", "O2", 
                  "P8", "T8", "FC6", "F4", 
                  "F8", "AF4"]
    
    info = mne.create_info(ch_names = ch_names,
                           sfreq = 128,
                           ch_types = 'eeg')
    
    # Construct Epochs
    epochs = mne.EpochsArray(data_arr, info)
    
    #epochs.metadata = metadata_df
    
    # Save Epochs
    
    return epochs