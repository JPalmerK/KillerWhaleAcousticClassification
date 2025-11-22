# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:58:40 2024

Script containing functions and methods for building H5 databases,
building model arechetectur, training and evaluaiton of models


@author: kaity
"""


import librosa
import librosa.display
import librosa.feature
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Activation
from keras.models import load_model
import keras
from keras import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.signal import butter, lfilter
#import gcsfs
from tensorflow.keras.callbacks import TensorBoard
#import seaborn as sns
from pathlib import Path

def bandpass(sig, rate, fmin, fmax, order=5):
    """
    Apply a bandpass filter to the input signal. Taken from birdnet
    https://github.com/birdnet-team/BirdNET-Analyzer/blob/main/birdnet_analyzer/audio.py

    Args:
        sig (numpy.ndarray): The input signal to be filtered.
        rate (int): The sampling rate of the input signal.
        fmin (float): The minimum frequency for the bandpass filter.
        fmax (float): The maximum frequency for the bandpass filter.
        order (int, optional): The order of the filter. Default is 5.

    Returns:
        numpy.ndarray: The filtered signal as a float32 array.
    """
    # # Check if we have to bandpass at all
    # if (fmin == cfg.SIG_FMIN and fmax == cfg.SIG_FMAX) or fmin > fmax:
    #     return sig

    # from scipy.signal import butter, lfilter

    # nyquist = 0.5 * rate

    # # Highpass?
    # if fmin > cfg.SIG_FMIN and fmax == cfg.SIG_FMAX:
    #     low = fmin / nyquist
    #     b, a = butter(order, low, btype="high")
    #     sig = lfilter(b, a, sig)

    # # Lowpass?
    # elif fmin == cfg.SIG_FMIN and fmax < cfg.SIG_FMAX:
    #     high = fmax / nyquist
    #     b, a = butter(order, high, btype="low")
    #     sig = lfilter(b, a, sig)

    # # Bandpass?
    # elif fmin > cfg.SIG_FMIN and fmax < cfg.SIG_FMAX:
    #     low = fmin / nyquist
    #     high = fmax / nyquist
    #     b, a = butter(order, [low, high], btype="band")
    #     sig = lfilter(b, a, sig)
    
    nyquist = rate/2 
    
    low = fmin / nyquist
    high = fmax / nyquist
    b, a = butter(order, [low, high], btype="band")
    sig = lfilter(b, a, sig)

    return sig.astype("float32")



def create_spectrogram(audio, return_snr=False,**kwargs):
    """
    Create the audio representation.

    kwargs options:
        clipDur (float): clip duration (sec).
        nfft (int): FFT window size (samples).
        hop_length (int): Hop length (samples).
        outSR (int): Target sampling rate.
        spec_type (str): Spectrogram type, 'normal' or 'mel'.
        spec_power(int): Magnitude spectrum type, approximate GPL, default =2
        min_freq (int): Minimum frequency to retain.
        rowNorm (bool): Normalize the spectrogram rows.
        colNorm (bool): Normalize the spectrogram columns.
        rmDCoffset (bool): Remove DC offset by subtracting mean.
        inSR (int): Original sample rate of the audio file (if decimation is needed).
        

    Returns:
        spectrogram (numpy.ndarray): Normalized spectrogram of the audio segment.
    """
    
    # Default parameters
    params = {
        'clipDur': 2,           # clip duration (sec)
        'nfft': 512,            # default FFT window size (samples)
        'hop_length': 25,     # default hop length (samples)
        'outSR': 16000,         # default target sampling rate
        'spec_type': 'normal',  # default spectrogram type, 'normal' or 'mel'
        'min_freq': None,       # default minimum frequency to retain
        'rowNorm': False,        # normalize the spectrogram rows
        'colNorm': False,        # normalize the spectrogram columns
        'rmDCoffset': True,     # remove DC offset by subtracting mean
        'NormalizeAudio':True,
        'inSR': None,            # original sample rate of the audio file
        'spec_power':2,
        'returnDB':True,         # return spectrogram in linear or convert to db 
        'PCEN':False,             # Per channel energy normalization
        'PCEN_power':0.5,
        'time_constant':0.4,
        'eps':1e-6,
        'gain':0.9,
        'power':0.5,
        'bias':2,
        'fmin':0,
        'fmax':16000,
        'NormalizeAudio': False, # normalized the audio to 0 to 1
        'Scale Spectrogram': False, # scale the spectrogram between 0 and 1
        'AnnotationTrain': 'bla',
        'AnnotationsTest': 'bla',
        'AnnotationsVal': 'bla',
        'Notes' : 'Balanced humpbacks by removing a bunch of humpbacks randomly'+
        'using batch norm and Raven parameters with Mel Spectrograms and PCEN '
    }

    # Update parameters based on kwargs
    for key, value in kwargs.items():
        if key in params:
            params[key] = value
        else:
            raise TypeError(f"Invalid keyword argument '{key}'")
    
    # Downsample data if necessary
    if params['inSR'] and params['inSR'] != params['outSR']:
        audio = librosa.resample(audio, orig_sr=params['inSR'], target_sr=params['outSR'])
    
    # Remove DC offset
    if params['rmDCoffset']:
        audio = audio - np.mean(audio)
        
        
    # Remove DC offset- must scale audio to 0-1 for PECN
    if params['NormalizeAudio']:
        audio = (audio - np.min(audio)) / (np.max(audio) - np.min(audio))  
    
    # Compute spectrogram
    if params['spec_type'] == 'normal':
        
        # If no power level specified set to 2 
        if 'spec_power' not in params:
            params['spec_power']=2
            
        spectrogram = np.abs(librosa.stft(audio, 
                                          n_fft=params['nfft'], 
                                          hop_length=params['hop_length']))
        
        # Normalize the spectrogram
        if params['rowNorm']==True:
            row_medians = np.median(spectrogram, axis=1, keepdims=True)
            spectrogram = spectrogram - row_medians
            
        if params['colNorm']==True:
            col_medians = np.median(spectrogram, axis=0, keepdims=True)
            spectrogram = spectrogram - col_medians
            
        spectrogram = np.abs(spectrogram)**params['spec_power']
        #spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
        if params['returnDB']==True:
            spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
    elif params['spec_type'] == 'mel':
        
        melspec = librosa.feature.melspectrogram(y=audio, 
                                                     sr=params['outSR'], 
                                                     n_fft=params['nfft'], 
                                                     hop_length=params['hop_length'],
                                                     fmin =params['fmin'],
                                                     power =params['spec_power'])
        
        # PCEN
        if params['PCEN']==True:
            
            spectrogram =librosa.pcen( 
                S = melspec * (2 ** params['PCEN_power']),
                time_constant=params['time_constant'],
                eps=params['eps'],
                gain =params['gain'],
                power=params['power'],
                bias=params['bias'],
                sr=params['outSR'],
                hop_length=params['hop_length'])
        else:
            spectrogram =melspec
            

        
        # Normalize the spectrogram
        if params['rowNorm']:
            row_medians = np.median(spectrogram, axis=1, keepdims=True)
            spectrogram = spectrogram - row_medians
            
        if params['colNorm']:
            col_medians = np.median(spectrogram, axis=0, keepdims=True)
            spectrogram = spectrogram - col_medians        
        
        if params['returnDB']==True:
            #spectrogram = np.round(spectrogram,2)
            spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    else:
        raise ValueError("Invalid spectrogram type. Supported types are 'normal' and 'mel'.")
    

        
    if params['Scale Spectrogram'] == True:
        spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram) + 1e-8)
        #plt.imshow(np.flipud(spectrogram))
        spectrogram =  np.float32(spectrogram)
        
    # Calculate SNR if requested
    if return_snr:
        signal_level = np.percentile(spectrogram, 85)
        noise_level = np.percentile(spectrogram, 25)
        SNR = signal_level - noise_level
        return spectrogram, SNR
    
    
    return spectrogram

# Redefine to independnetly create the audio representations
def load_and_process_audio_segment(file_path, start_time, end_time,
                                   return_snr=False, **kwargs):
    """
    Load an audio segment from a file, process it, and create a spectrogram image.

    Parameters:
        file_path (str): Path to the audio file.
        start_time (float): Start time of the audio segment in seconds.
        end_time (float): End time of the audio segment in seconds.
        return_snr (bool): Flag to return Signal-to-Noise Ratio of the spectrogram.
        **kwargs: Additional keyword arguments passed to create_spectrogram.

    Returns:
        spectrogram (numpy.ndarray): Normalized spectrogram of the audio segment.
        SNR (float): Signal-to-Noise Ratio of the spectrogram (if return_snr=True).
    """
    
    # Default parameters for create_spectrogram
    params = {
        'clipDur': 2,           # clip duration (sec)
        'outSR': 16000,         # default target sampling rate
    }

    # Update parameters based on kwargs
    for key, value in kwargs.items():
        if key in params:
            params[key] = value
        #else:
            #raise TypeError(f"Invalid keyword argument '{key}'")

    # Get the duration of the audio file
    file_duration = librosa.get_duration(path=file_path)
    
    # Calculate the duration of the desired audio segment
    #duration = end_time - start_time
    
    # Calculate the center time of the desired clip
    center_time = (start_time + end_time) / 2.0
    
    # Calculate new start and end times based on the center and clip duration
    new_start_time = center_time - params['clipDur'] / 2
    new_end_time = center_time + params['clipDur'] / 2
    
    # Adjust start and end times if the clip duration is less than desired
    if new_end_time - new_start_time < params['clipDur']:
        pad_length = params['clipDur'] - (new_end_time - new_start_time)
        new_start_time = max(0, new_start_time - pad_length / 2.0)
        new_end_time = min(file_duration, new_end_time + pad_length / 2.0)
    
    # Ensure start and end times don't exceed the bounds of the audio file
    new_start_time = max(0, min(new_start_time, file_duration - params['clipDur']))
    new_end_time = max(params['clipDur'], min(new_end_time, file_duration))
    
    # Load audio segment and downsample to the defined sampling rate
    audio_data, sample_rate = librosa.load(file_path, 
                                           sr=params['outSR'], 
                                           offset=new_start_time,
                                           duration=params['clipDur'],
                                           mono=False)
    # Determine the number of channels
    num_channels = audio_data.shape[0] if audio_data.ndim > 1 else 1
    
    # Retain only the first channel if there are multiple
    if num_channels > 1:
        print(f"Audio has {num_channels} channels. Retaining only the first channel.")
        audio_data = audio_data[0]

    
    # Create audio representation
    spec, snr = create_spectrogram(audio_data, return_snr=return_snr, **kwargs)

    if return_snr:
        return spec, snr
    else:
        return spec

# Load and process audio segments, and save spectrograms and labels to HDF5 file
def create_hdf5_dataset(annotations, hdf5_filename, parms):
    """
    Create an HDF5 database with spectrogram images for DCLDE data.
    
    Parameters:
        annotations (pd.df): pandas dataframe with headers for 'FilePath', 
        'FileBeginSec','FileEndSec', 'label','traintest', 'Dep', 'Provider',
        'KW','KW_certain'
    Returns:
        spec_normalized (numpy.ndarray): Normalized spectrogram of the audio segment.
        SNR (float): Signal-to-Noise Ratio of the spectrogram.
    """
    # write the parameters for recovering latter
          # Store the parameters as attributes
    # Open HDF5 file in write mode
    with h5py.File(hdf5_filename, 'w') as hf:
        # Store parameters as attributes
        for key, value in parms.items():
            if value is not None:
                hf.attrs[key] = value
        
        
        # Create groups for train and test datasets
        train_group = hf.create_group('train')
        test_group = hf.create_group('test')

        # Iterate through annotations and create datasets
        for idx, row in annotations.iloc[0:].iterrows():
       # for idx, row in annotations.iterrows():
            file_path = row['FilePath']
            start_time = row['FileBeginSec']
            end_time = row['FileEndSec']
            label = row['label']
            traintest = row['traintest']
            dep = row['Dep']
            provider = row['Provider']
            kw = row['KW']
            kwCertin = row['KW_certain']
            utc = row['UTC']

            # Load and process audio segment
            spec, SNR = load_and_process_audio_segment(file_path, start_time, 
                                                       end_time, 
                                                       return_snr=True, **parms)

            # Determine which group to store in (train or test)
            dataset = train_group if traintest == 'Train' else test_group

            # Create datasets for each attribute
            data_group = dataset.create_group(f'{idx}')
            data_group.create_dataset('spectrogram', data=spec)
            data_group.create_dataset('label', data=label)
            data_group.create_dataset('deployment', data=dep)
            data_group.create_dataset('provider', data=provider)
            data_group.create_dataset('KW', data=kw)
            data_group.create_dataset('KW_certain', data=kwCertin)
            data_group.create_dataset('SNR', data=SNR)
            data_group.create_dataset('UTC', data=utc)

            print(f"Processed {idx + 1} of {len(annotations)}")


#############################################
# Create HDF5 with parallelization
################################################


from concurrent.futures import ProcessPoolExecutor

def process_audio_segment(row, parms):
    """
    Process a single audio segment and return the results.
    
    Parameters:
        row (pd.Series): A row from the annotations DataFrame.
        parms (dict): Parameters for processing the audio.
        
    Returns:
        dict: A dictionary containing the processed results.
    """
    try:
        file_path = row['FilePath']
        start_time = row['FileBeginSec']
        end_time = row['FileEndSec']
        label = row['label']
        traintest = row['traintest']
        dep = row['Dep']
        provider = row['Provider']
        kw = row['KW']
        kwCertin = row['KW_certain']
        utc = row['UTC']

        # Load and process audio segment
        spec, SNR = load_and_process_audio_segment(file_path, start_time, end_time, return_snr=True, **parms)

        return {
            'traintest': traintest,
            'data': {
                'spectrogram': spec,
                'label': label,
                'deployment': dep,
                'provider': provider,
                'KW': kw,
                'KW_certain': kwCertin,
                'SNR': SNR,
                'UTC': utc
            }
        }
    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        return None

def process_audio_segment_wrapper(args):
    """
    Wrapper for process_audio_segment to allow passing arguments for multiprocessing.
    
    Parameters:
        args (tuple): A tuple containing (row, parms).
        
    Returns:
        dict: Processed results from process_audio_segment.
    """
    row, parms = args
    return process_audio_segment(row, parms)

def process_batch(batch, parms):
    """
    Process a batch of annotations in parallel.
    
    Parameters:
        batch (pd.DataFrame): A batch of annotations.
        parms (dict): Parameters for processing the audio.
        
    Returns:
        list: Processed results for the batch.
    """
    with ProcessPoolExecutor() as executor:
        # Pass rows and parameters as tuples to the wrapper function
        results = list(executor.map(process_audio_segment_wrapper, [(row, parms) for _, row in batch.iterrows()]))
    return list(filter(None, results))  # Exclude failed processes (None values)

def create_hdf5_dataset_parallel2(annotations, hdf5_filename, parms, batch_size=100):
    """
    Create an HDF5 database with spectrogram images for DCLDE data using batch processing.
    
    Parameters:
        annotations (pd.DataFrame): Annotations DataFrame.
        hdf5_filename (str): Output HDF5 file name.
        parms (dict): Parameters for processing the audio.
        batch_size (int): Number of rows to process in each batch.
    """
    with h5py.File(hdf5_filename, 'w') as hf:
        # Store parameters as attributes
        for key, value in parms.items():
            if value is not None:
                hf.attrs[key] = value

        # Create groups for train and test datasets
        train_group = hf.create_group('train')
        test_group = hf.create_group('test')

        # Process annotations in batches
        num_batches = int(np.ceil(len(annotations) / batch_size))
        for batch_idx in range(num_batches):
            print(f"Processing batch {batch_idx + 1}/{num_batches}...")
            batch = annotations.iloc[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_results = process_batch(batch, parms)

            # Write batch results to HDF5 file
            for result in batch_results:
                dataset = train_group if result['traintest'] == 'Train' else test_group
                group_idx = len(dataset)  # Determine next available group index
                data_group = dataset.create_group(f"{group_idx}")

                for key, value in result['data'].items():
                    data_group.create_dataset(key, data=value)

            print(f"Completed batch {batch_idx + 1}/{num_batches}")

    print("HDF5 dataset creation completed.")


def create_clip_folder(annotations, fileOutLoc, parms):
    """
    Create folders wtih clips of each label
    """
    # Iterate through annotations and create datasets
    #for idx, row in annotations.iloc[80816:].iterrows():
    for idx, row in annotations.iterrows():
        file_path = row['FilePath']
        start_time = row['FileBeginSec']
        end_time = row['FileEndSec']
        #label = row['label']
        #traintest = row['traintest']
        #dep = row['Dep']
        #provider = row['Provider']
        #kw = row['KW']
        #kwCertin = row['KW_certain']
        #utc = row['UTC']

        # Load and process audio segment
        spec, SNR = load_and_process_audio_segment(file_path, start_time, 
                                                   end_time, return_snr=True,
                                                   **parms)    


import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define your functions load_and_process_audio_segment() and create_hdf5_dataset() here

# Function to process each row in annotations DataFrame
def process_row(row):
    file_path = row['FilePath']
    start_time = row['FileBeginSec']
    end_time = row['FileEndSec']
    label = row['label']
    traintest = row['traintest']
    dep = row['Dep']
    provider = row['Provider']
    kw = row['KW']
    kwCertin = row['KW_certain']
    

    spec, SNR = load_and_process_audio_segment(file_path, start_time, end_time)

    return spec, SNR, traintest, label, dep, provider, kw, kwCertin

# Main function to create HDF5 dataset with multithreading
def create_hdf5_dataset_parallel(annotations, hdf5_filename, num_threads=6):
    with h5py.File(hdf5_filename, 'w') as hf:
        train_group = hf.create_group('train')
        test_group = hf.create_group('test')

        n_rows = len(annotations)

        # Use ThreadPoolExecutor to parallelize the processing
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_row, annotations.iloc[ii]) for ii in range(n_rows)]

            for future in as_completed(futures):
                spec, SNR, traintest, label, dep, provider, kw, kwCertin = future.result()

                if traintest == 'Train':  # 80% train, 20% test
                    dataset = train_group
                else:
                    dataset = test_group

                ii = len(dataset)  # Get the index for the new dataset

                dataset.create_dataset(f'{ii}/spectrogram', data=spec)
                dataset.create_dataset(f'{ii}/label', data=label)
                dataset.create_dataset(f'{ii}/deployment', data=dep)
                dataset.create_dataset(f'{ii}/provider', data=provider)
                dataset.create_dataset(f'{ii}/KW', data=kw)
                dataset.create_dataset(f'{ii}/KW_certain', data=kwCertin)
                dataset.create_dataset(f'{ii}/SNR', data=SNR)
                

                print(ii, ' of ', len(annotations))

    
class BatchLoader2(keras.utils.Sequence):
    def __init__(self, hdf5_file, batch_size=250, trainTest='train',
                 shuffle=True, n_classes=7, return_data_labels=False,
                 minFreq = None):
        self.hf = h5py.File(hdf5_file, 'r')
        self.batch_size = batch_size
        self.trainTest = trainTest
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.return_data_labels=return_data_labels
        self.data_keys = list(self.hf[trainTest].keys())
        self.num_samples = len(self.data_keys)
        self.indexes = np.arange(self.num_samples)
        
        # Get th spectrogram size, assume something in Train
        self.train_group = self.hf[trainTest]
        self.first_key = list(self.train_group.keys())[0]
        self.data =self.train_group[self.first_key]['spectrogram']
        self.specSize = self.data.shape
        
        # If a frequency limit is set then  figure out what that is now
        self.minFreq = minFreq
        self.minIdx = 0
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, index):
        start_index = (index-1) * self.batch_size
        end_index = min((index ) * self.batch_size, self.num_samples)-1
        
        batch_data = []
        batch_labels = []
        
        for i in range(start_index, end_index):
            key = self.data_keys[self.indexes[i]]
            spec = self.hf[self.trainTest][key]['spectrogram'][self.minIdx:,:]
            label = self.hf[self.trainTest][key]['label'][()]
            batch_data.append(spec)
            batch_labels.append(label)
        
        if self.return_data_labels:
            return batch_data, batch_labels
        else:
            return np.array(batch_data), keras.utils.to_categorical(np.array(batch_labels),  num_classes=self.n_classes)

    
    def __shuffle__(self):
        np.random.shuffle(self.indexes)
        print('shuffled!')
    
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            print('Epoc end all shuffled!')

import numpy as np
import keras
import random



def train_model_history(model, train_generator, val_generator, epochs, tensorBoard=False):
    '''
    Train model function (same as train_model()) but preserves history
    

    Parameters
    ----------
    model : keras model
        trained Keras model.
    train_generator : 
        train batch generator created with BatchLoader2 .
    val_generator : TYPE
        train batch generator created with BatchLoader2 .
    epochs : int
        Number of epocs to run.
    tensorBoard : bool, optional (False)
        Whether to try to run tensorboard. Not presently working on windows 
        machine

    Returns
    -------
    '''
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5, 
                                   restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
    
    if tensorBoard:
        # Define the TensorBoard callback
        tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
        
        # Train the model and capture history
        history = model.fit(x=train_generator,
                            epochs=epochs,
                            validation_data=val_generator,
                            callbacks=[early_stopping, tensorboard_callback])
    else:
        # Train the model and capture history
        history = model.fit(x=train_generator,
                            epochs=epochs,
                            validation_data=val_generator,
                            callbacks=[early_stopping, ReduceLROnPlateau])

    return history    


class BatchLoader_hardNegs(keras.utils.Sequence):
    def __init__(self, hdf5_file, batch_size=250, trainTest='train',
                 shuffle=True, n_classes=7, return_data_labels=False,
                 minFreq=None):
        self.hf = h5py.File(hdf5_file, 'r')
        self.batch_size = batch_size
        self.trainTest = trainTest
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.return_data_labels = return_data_labels

        # Store references to both train and test groups
        self.train_group = self.hf['train']
        self.test_group = self.hf['test']

        # Initialize data keys and their source group (train or test)
        self.data_keys = list(self.hf[trainTest].keys())
        self.key_source = {key: trainTest for key in self.data_keys}  # Track which group a key belongs to
        self.num_samples = len(self.data_keys)
        self.indexes = np.arange(self.num_samples)

        # Get spectrogram size from the first sample
        self.first_key = self.data_keys[0]
        self.specSize = self.hf[trainTest][self.first_key]['spectrogram'].shape

        # Frequency settings
        self.minFreq = minFreq
        self.minIdx = 0

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, self.num_samples)

        batch_data = []
        batch_labels = []

        for i in range(start_index, end_index):
            key = self.data_keys[self.indexes[i]]
            source_group = self.key_source[key]  # Determine whether it came from train or test

            spec = self.hf[source_group][key]['spectrogram'][self.minIdx:, :]
            label = self.hf[source_group][key]['label'][()]
            batch_data.append(spec)
            batch_labels.append(label)

        if self.return_data_labels:
            return batch_data, batch_labels
        else:
            return np.array(batch_data), keras.utils.to_categorical(np.array(batch_labels), num_classes=self.n_classes)

    def add_hard_negatives(self, hard_negatives):
        """Move hard negatives from test to train and update indices."""
        new_keys = [key for key in hard_negatives if key in self.test_group]  # Ensure they exist in test

        if not new_keys:
            print("Warning: No valid hard negatives found in test.")
            return

        # Add new keys and update their source to 'test'
        for key in new_keys:
            self.key_source[key] = 'test'  # Even if they're now in training, they originated from test

        self.data_keys = list(np.unique(self.data_keys + new_keys))  # Merge and remove duplicates
        self.num_samples = len(self.data_keys)
        self.indexes = np.arange(self.num_samples)  # Update indexes

        self.__shuffle__()
        print(f"Training set now has {self.num_samples} samples (including hard negatives).")

    def __shuffle__(self):
        np.random.shuffle(self.indexes)
        print('Shuffled!')

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            print('Epoch end: all shuffled!')


def train_model_history_hardNeg(model, train_generator,
                                val_generator, 
                                epochs, tensorBoard=False):
    '''
    Train model function with hard negative mining.
    
    Parameters
    ----------
    model : keras model
        trained Keras model.
    train_generator : 
        train batch generator created with BatchLoader2 .
    val_generator : 
        validation batch generator created with BatchLoader2 .
    epochs : int
        Number of epochs to run.
    tensorBoard : bool, optional
        Whether to run tensorboard.

    Returns
    -------
    history, history_hard
    '''
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3,
                                   restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)
    callbacks = [early_stopping, reduce_lr]

    # Train model initially
    print("Starting initial training...")
    history = model.fit(x=train_generator, epochs=epochs,
                        validation_data=val_generator,
                        callbacks=callbacks)  

    # Identify hard negatives
    print("Identifying hard negatives...")
    hard_negatives = identify_hard_negatives(model, val_generator)
    train_generator.add_hard_negatives(hard_negatives)  # Add them to training
    
    # Define a new early stopping for hard negatives training
    early_stopping_hard = EarlyStopping(monitor='val_loss', patience=3,
                                        restore_best_weights=True)
    callbacks_hard = [early_stopping_hard, reduce_lr]
    
    # Retrain model with hard negatives
    print("Retraining with hard negatives...")
    history_hard = model.fit(x=train_generator, epochs=int(epochs // 2), 
                             validation_data=val_generator, 
                             callbacks=callbacks_hard)


    return history, history_hard

def identify_hard_negatives(model, data_generator):
    '''
    Identify hard negatives by running inference and checking misclassifications.
    
    Parameters
    ----------
    model : keras model
        Trained model.
    data_generator : keras.utils.Sequence
        Validation generator.
    
    Returns
    -------
    hard_negatives : list
        List of misclassified sample keys.
    '''
    hard_negatives = []

    for i in range(len(data_generator)):
        batch_data, batch_labels = data_generator[i]
        predictions = model.predict(batch_data)

        for j, (pred, true_label) in enumerate(zip(predictions, batch_labels)):
            if np.argmax(pred) != np.argmax(true_label):  # Misclassification
                sample_key = data_generator.data_keys[i * data_generator.batch_size + j]
                if sample_key not in hard_negatives:  # Avoid duplicates
                    hard_negatives.append(sample_key)

    print(f"Found {len(hard_negatives)} hard negatives.")
    return hard_negatives

def plot_training_curves(history):
    '''
    Plots training and validation accuracy/loss curves. See also 


    Parameters
    ----------
    history : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    # Extract data from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot accuracy
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    

def check_spectrogram_dimensions(hdf5_file):
    with h5py.File(hdf5_file, 'r') as hf:
        spectrogram_shapes = set()  # Set to store unique spectrogram shapes
        for group_name in hf:
            group = hf[group_name]
            for key in group:
                spectrograms = group[key]['spectrogram'][:]
                for spectrogram in spectrograms:
                    spectrogram_shapes.add(spectrogram.shape)
    return spectrogram_shapes

###########################################################################
# Different Models
#########################################################################

from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D
from tensorflow import keras
from tensorflow.keras import layers

def create_resnet50(input_shape, num_classes):
    # Use a custom input tensor
    input_tensor = Input(shape=input_shape)
    # Create the base model with the modified input tensor
    base_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)
    #base_model.summary()
    
    base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape)
    # Add global average pooling to reduce feature map dimensions
    x = GlobalAveragePooling2D()(base_model.output)
    
    # Add a fully connected layer for classification
    output = Dense(num_classes, activation='softmax')(x)
    
    # Create the new model
    model = Model(inputs=base_model.input, outputs=output)
    #model.summary()
    return model

def create_resnet101(input_shape, num_classes):
    # Use a custom input tensor
    input_tensor = Input(shape=input_shape)
    # Create the base model with the modified input tensor
    base_model = keras.applications.ResNet101(include_top=False, weights=None, input_tensor=input_tensor)
    #base_model.summary()
    
    base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape)
    # Add global average pooling to reduce feature map dimensions
    x = GlobalAveragePooling2D()(base_model.output)
    
    # Add a fully connected layer for classification
    output = Dense(num_classes, activation='softmax')(x)
    
    # Create the new model
    model = Model(inputs=base_model.input, outputs=output)
    model.summary()
    return model

def create_model_with_resnet(input_shape, num_classes, actName = 'relu'):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation=actName)(input_layer)
    #x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation=actName)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Add three identity blocks
    x = identity_block(x, 64)
    x = identity_block(x, 64)
    x = identity_block(x, 64)
    
    x = Flatten()(x)
    x = Dense(128, activation=actName)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def create_wider_model(input_shape, num_classes, actName='relu'):
    input_layer = Input(shape=input_shape)
    
    x = Conv2D(64, kernel_size=(3, 3), activation=actName)(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(128, kernel_size=(3, 3), activation=actName)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Same identity blocks
    x = identity_block(x, 128)
    x = identity_block(x, 128)
    x = identity_block(x, 128)
    
    x = Flatten()(x)
    x = Dense(256, activation=actName)(x)
    
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def identity_block(input_tensor, filters):
    x = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
    #x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Add()([x, input_tensor])  # Add the input tensor to the output of the second convolution
    x = Activation('relu')(x)
    return x

def create_wider_model2(input_shape, num_classes, actName='relu'):
    input_layer = Input(shape=input_shape)
    
    x = Conv2D(64, kernel_size=(3, 3), activation=None, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation(actName)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(64, kernel_size=(3, 3), activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(actName)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Same identity blocks
    x = identity_block2(x, 64)
    x = identity_block2(x, 64)
    x = identity_block2(x, 64)
    
    x = Flatten()(x)
    x = Dense(128, activation=actName)(x)
    
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def identity_block2(input_tensor, filters):
    x = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Shortcut connection
    shortcut = Conv2D(filters, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

# Enable mixed precision for speedup on supported hardware

def create_birdnet_like_model(input_shape=(128, 883, 1), num_classes=2):
    inputs = keras.Input(shape=input_shape)
    
    # Initial Conv Layer
    x = layers.Conv2D(32, (3, 3), strides=1, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Depthwise Separable Convolutions (efficient feature extraction)
    for filters in [64, 128, 256]:
        x = layers.SeparableConv2D(filters, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
    
    # Bottleneck Conv Layer
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Fully Connected Layer
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Output Layer
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)  # Ensure float32 output
    
    model = keras.Model(inputs, outputs, name="BirdNET_Like_Model")
    return model

# Compile model
def compile_model(model, loss_val = 'categorical_crossentropy'):
    model.compile(loss=loss_val,
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model


import tensorflow as tf
from tensorflow.keras import models, regularizers

def residual_block(x, filters):
    """ Residual block: Conv + BN + ReLU + Conv + BN + Add """
    shortcut = x  # Identity mapping
    x = layers.Conv2D(filters, (3, 3), padding="same", activation=None, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", activation=None, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])  # Residual connection
    x = layers.ReLU()(x)
    return x

def downsampling_block(x, filters):
    """ Downsampling block: Conv (stride=2) + BN + ReLU """
    x = layers.Conv2D(filters, (3, 3), strides=2, padding="same", activation=None, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def create_birdnet(input_shape=(64, 384, 1), num_classes=987):
    inputs = layers.Input(shape=input_shape)

    # Preprocessing Conv Block
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)

    # ResStack 1
    x = downsampling_block(x, 64)   # (64, 32, 96)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # ResStack 2
    x = downsampling_block(x, 128)  # (128, 16, 48)
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    # ResStack 3
    x = downsampling_block(x, 256)  # (256, 8, 24)
    x = residual_block(x, 256)
    x = residual_block(x, 256)

    # ResStack 4
    x = downsampling_block(x, 512)  # (512, 4, 12)
    x = residual_block(x, 512)
    x = residual_block(x, 512)

    # Classification Head
    x = layers.Conv2D(512, (4, 10), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(1024, (1, 1), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(num_classes, (1, 1), padding="same", activation=None, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Global Log-Mel Energy Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Output Layer with Sigmoid Activation
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)  

    model = models.Model(inputs, outputs, name="BirdNET")
    return model

# Compile model
def compile_model(model):
    model.compile(
        loss='binary_crossentropy',  # Multi-label classification
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  
        metrics=['accuracy']
    )
    return model



#############################################################################
# Bigger resenet
#############################################################################
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import  ReLU


def ConvBlock(inputs, filters, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def IdentityBlock(inputs, filters, kernel_size, padding='same'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return Add()([x, inputs])



def ResNet1_testing(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
  
    # BLOCK-1
    x = ConvBlock(input_layer, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

   
    # FINAL BLOCK
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)



    return model


def ResNet18(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
  
    # BLOCK-1
    x = ConvBlock(input_layer, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # BLOCK-2
    x = ConvBlock(x, filters=64, kernel_size=(3, 3), padding='same')
    x = ConvBlock(x, filters=64, kernel_size=(3, 3), padding='same')
    x = Dropout(0.5)(x)
    op2_1 = IdentityBlock(x, filters=64, kernel_size=(3, 3))

    x = ConvBlock(op2_1, filters=64, kernel_size=(3, 3), padding='same')
    x = ConvBlock(x, filters=64, kernel_size=(3, 3), padding='same')
    x = Dropout(0.5)(x)
    op2 = IdentityBlock(x, filters=64, kernel_size=(3, 3))

    # BLOCK-3
    x = ConvBlock(op2, filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')
    x = ConvBlock(x, filters=128, kernel_size=(3, 3), padding='same')
    adjust_op2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding='valid')(op2)
    x = Dropout(0.5)(x)
    op3_1 = IdentityBlock(x, filters=128, kernel_size=(3, 3))

    x = ConvBlock(op3_1, filters=128, kernel_size=(3, 3), padding='same')
    x = ConvBlock(x, filters=128, kernel_size=(3, 3), padding='same')
    x = Dropout(0.5)(x)
    op3 = IdentityBlock(x, filters=128, kernel_size=(3, 3))

    # BLOCK-4
    x = ConvBlock(op3, filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')
    x = ConvBlock(x, filters=256, kernel_size=(3, 3), padding='same')
    adjust_op3 = Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2), padding='valid')(op3)
    x = Dropout(0.5)(x)
    op4_1 = IdentityBlock(x, filters=256, kernel_size=(3, 3))

    x = ConvBlock(op4_1, filters=256, kernel_size=(3, 3), padding='same')
    x = ConvBlock(x, filters=256, kernel_size=(3, 3), padding='same')
    x = Dropout(0.5)(x)
    op4 = IdentityBlock(x, filters=256, kernel_size=(3, 3))

    # BLOCK-5
    x = ConvBlock(op4, filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')
    x = ConvBlock(x, filters=512, kernel_size=(3, 3), padding='same')
    adjust_op4 = Conv2D(filters=512, kernel_size=(1, 1), strides=(2, 2), padding='valid')(op4)
    x = Dropout(0.5)(x)
    op5_1 = IdentityBlock(x, filters=512, kernel_size=(3, 3))

    x = ConvBlock(op5_1, filters=512, kernel_size=(3, 3), padding='same')
    x = ConvBlock(x, filters=512, kernel_size=(3, 3), padding='same')
    x = Dropout(0.5)(x)
    op5 = IdentityBlock(x, filters=512, kernel_size=(3, 3))

    # FINAL BLOCK
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)



    return model

def ResNet18_batchNorm(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
  
    # BLOCK-1
    x = ConvBlock(input_layer, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # BLOCK-2
    x = ConvBlock(x, filters=64, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = ConvBlock(x, filters=64, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = Dropout(0.5)(x)
    op2_1 = IdentityBlock(x, filters=64, kernel_size=(3, 3))

    x = ConvBlock(op2_1, filters=64, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = ConvBlock(x, filters=64, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = Dropout(0.5)(x)
    op2 = IdentityBlock(x, filters=64, kernel_size=(3, 3))

    # BLOCK-3
    x = ConvBlock(op2, filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = ConvBlock(x, filters=128, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    adjust_op2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding='valid')(op2)
    x = Dropout(0.5)(x)
    op3_1 = IdentityBlock(x, filters=128, kernel_size=(3, 3))

    x = ConvBlock(op3_1, filters=128, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = ConvBlock(x, filters=128, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = Dropout(0.5)(x)
    op3 = IdentityBlock(x, filters=128, kernel_size=(3, 3))

    # BLOCK-4
    x = ConvBlock(op3, filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = ConvBlock(x, filters=256, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    adjust_op3 = Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2), padding='valid')(op3)
    x = Dropout(0.5)(x)
    op4_1 = IdentityBlock(x, filters=256, kernel_size=(3, 3))

    x = ConvBlock(op4_1, filters=256, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = ConvBlock(x, filters=256, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = Dropout(0.5)(x)
    op4 = IdentityBlock(x, filters=256, kernel_size=(3, 3))

    # BLOCK-5
    x = ConvBlock(op4, filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = ConvBlock(x, filters=512, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    adjust_op4 = Conv2D(filters=512, kernel_size=(1, 1), strides=(2, 2), padding='valid')(op4)
    x = Dropout(0.5)(x)
    op5_1 = IdentityBlock(x, filters=512, kernel_size=(3, 3))

    x = ConvBlock(op5_1, filters=512, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = ConvBlock(x, filters=512, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = Dropout(0.5)(x)
    op5 = IdentityBlock(x, filters=512, kernel_size=(3, 3))

    # FINAL BLOCK
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)  # Optional Batch Normalization for Dense layer
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

import json    
def saveModelData(model,modelName, savePath, metadata):
    '''
    Function to save the model with the associated metadata
    
    Parameters
    ----------
    model : keras model
    modelName : string
        name of the model excluding keras
    savePath : string
        Full path file for save model location including model name
    metadata : dictionary
        dictionary containing the parameters used for training excluding
        the hdf5 files for training and validation

    Returns
    -------
    None.
    
    Example :
        metadata = {
            "h5TrainTest": "spectrogram_8kHz_norm01_fft256_hop128.h5",
            "h5TrainEval": "spectrogram_8kHz_norm01_fft256_hop128.h5",
            "parameters": {
                "epochs": 20,
                "batch_size": 32,
                "optimizer": "adam"
            }
        }

    '''
    model.save(savePath + modelName + '.keras')
    with open(savePath + modelName + '_metadata.json', 'w') as f:
        json.dump(metadata, f)
        
#########################################################################
# Model Evaluation Class
#########################################################################

import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# So this is well and good but I'd like to create be able to look at the timestamps
# from the predictions so we can do something lie softmax

def plot_training_curves(history, titleStr = ' blart'):
    """
    Plots training and validation accuracy/loss curves.

    Parameters:
        history: History object returned by model.fit().
    """
    # Extract data from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot accuracy
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy'+ titleStr)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss'+titleStr)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()






#########################################################################
#Pipeline for producing streaming detections- Does not work on GS cloud so comment
##########################################################################


import soundfile as sf

from sklearn.metrics import precision_recall_curve, average_precision_score

class ModelEvaluator:
    def __init__(self, loaded_model, val_batch_loader, label_dict):
        """
        Class to evaluate the trained model on the validataion H5 file
        Initialize the ModelEvaluator object.
        
        Parameters
        ----------
            loaded_model : keras model
            The trained keras model to evaluate
            
            val_batch_loader : object
            Batch loader to feed spectrograms from the H5 model to the kerask 
            model for prediction. See BatchLoader2 method in EcotypeDefs

            label_dict: dictionary
            Dictionary mapping numeric labels used to train the model to 
            to human-readable labels. This should contain all labels in the
            origional training dataset E.g. 
            label_dict = dict(zip(annot_test['label'], annot_test['Labels']))
            
        Methods
        -------
        evaluate_model()
            Creates model predictions for all evaluation methods. Run first.
        confusion_matrix()
            Creates confusion matrix based on predicted scores and labels. 
        score_distributions()
            Creates violin plots of score distributions for true and false
            positive predictions
        """
        self.model = loaded_model
        self.val_batch_loader = val_batch_loader
        self.label_dict = label_dict
        self.y_true_accum = []
        self.y_pred_accum = []
        self.score_accum = []

    def evaluate_model(self):
        """Runs the model on the validation data and stores predictions and scores."""
        total_batches = len(self.val_batch_loader)
        for i in range(total_batches):
            batch_data = self.val_batch_loader.__getitem__(i)
            batch_scores = self.model.predict(np.asarray(batch_data[0]))  # Model outputs (softmax scores)
            batch_pred_labels = np.argmax(batch_scores, axis=1)
            batch_true_labels = batch_data[1]

            # Accumulate true labels, predicted labels, and scores
            self.y_true_accum.extend(batch_true_labels)
            self.y_pred_accum.extend(batch_pred_labels)
            self.score_accum.extend(batch_scores)

            print(f'Batch {i+1}/{total_batches} processed')

        self.y_true_accum = np.array(self.y_true_accum)
        self.y_pred_accum = np.array(self.y_pred_accum)
        self.score_accum = np.array(self.score_accum)

    def confusion_matrix(self):
        """Computes a confusion matrix with human-readable labels and accuracy."""
        conf_matrix_raw = confusion_matrix(self.y_true_accum, self.y_pred_accum)

        # Normalize confusion matrix by rows
        conf_matrix_percent = conf_matrix_raw.astype(np.float64)
        row_sums = conf_matrix_raw.sum(axis=1, keepdims=True)
        conf_matrix_percent = np.round(np.divide(conf_matrix_percent, row_sums, 
                                        where=row_sums != 0) * 100,2)

        # Map numeric labels to human-readable labels
        unique_labels = sorted(set(self.y_true_accum) | set(self.y_pred_accum))
        human_labels = [self.label_dict[label] for label in unique_labels]

        # Format confusion matrix to two decimal places
        conf_matrix_percent_formatted = np.array([[f"{value:.2f}" for value in row]
                                                  for row in conf_matrix_percent])

        # Create DataFrame
        conf_matrix_df = pd.DataFrame(conf_matrix_percent_formatted, index=human_labels, columns=human_labels)

        # Compute overall accuracy
        accuracy = accuracy_score(self.y_true_accum, self.y_pred_accum)

        return conf_matrix_df, conf_matrix_raw, accuracy

    def score_distributions(self):
        """Generates a DataFrame of score distributions for true positives and false positives."""
        score_data = []
        for i, true_label in enumerate(self.y_true_accum):
            pred_label = self.y_pred_accum[i]
            scores = self.score_accum[i]

            for class_label, score in enumerate(scores):
                label_type = "True Positive" if (true_label == class_label == pred_label) else "False Positive"
                score_data.append({
                    "Class": self.label_dict[class_label],
                    "Score": score,
                    "Type": label_type
                })

        score_df = pd.DataFrame(score_data)

        # Plot paired violin plot
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=score_df, x="Class", y="Score", hue="Type", split=True, inner="quartile", palette="muted")
        plt.title("Score Distributions for True Positives and False Positives")
        plt.xticks(rotation=45)
        plt.ylabel("Score")
        plt.xlabel("Class")
        plt.legend(title="Type")
        plt.tight_layout()
        plt.show()

        return score_df
    def precision_recall_curves(self):
        """Computes and plots precision-recall curves for all classes."""
        num_classes = self.score_accum.shape[1]
        precision_recall_data = {}
    
        plt.figure(figsize=(10, 8))
    
        # Calculate PR curves for each class
        for class_idx in range(num_classes):
            # Check if the class is present in the dataset
            class_present = (self.y_true_accum == class_idx).any()
    
            if not class_present:
                print(f"Class {self.label_dict[class_idx]} is not present in the validation dataset.")
                # Store empty results for missing class
                precision_recall_data[self.label_dict[class_idx]] = {
                    "precision": None,
                    "recall": None,
                    "average_precision": None
                }
                continue
    
            # Binarize true labels for the current class
            true_binary = (self.y_true_accum == class_idx).astype(int)
    
            # Retrieve scores for the current class
            class_scores = self.score_accum[:, class_idx]
    
            # Compute precision, recall, and average precision score
            precision, recall, _ = precision_recall_curve(true_binary, class_scores)
            avg_precision = average_precision_score(true_binary, class_scores)
    
            # Store the data
            precision_recall_data[self.label_dict[class_idx]] = {
                "precision": precision,
                "recall": recall,
                "average_precision": avg_precision
            }
    
            # Plot PR curve
            plt.plot(recall, precision, label=f"{self.label_dict[class_idx]} (AP={avg_precision:.2f})")
    
        # Finalize plot
        plt.title("Precision-Recall Curves")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="best")
        plt.grid()
        plt.tight_layout()
        plt.show()
    
        return precision_recall_data

    
import tensorflow.lite as tflite
import scipy.special


import os
import numpy as np
import pandas as pd
import librosa
import scipy.special
import scipy.signal
from tensorflow import lite as tflite  # Using TensorFlow's built-in lite interpreter


class BirdNetPredictor:
    def __init__(self, model_path, label_path, audio_folder, 
                 sample_rate=48000, 
                 audio_duration=3.0, 
                 confidence_thresh=0.5,):
        """
        Processor class for running TFLite (e.g. BirdNET) models on audio files
        in a folder.
        
        Parameters:
            model_path (str): Path to the TensorFlow Lite model.
            label_path (str): Path to the label file (text file with one label per line).
            audio_folder (str): Path to the folder containing audio files.
            sample_rate (int): Sample rate for the model (default 48000).
            audio_duration (float): Duration in seconds of audio to classify (default 3.0).
            confidence_thresh (float): Confidence threshold for filtering predictions.
        """
        self.model_path = model_path
        self.label_path = label_path
        self.audio_folder = audio_folder
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.confidence_thresh = confidence_thresh
        
        # Load model and labels using TensorFlow Lite interpreter
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Extract expected duration from model input shape
        input_shape = self.input_details[0]['shape']
        self.audio_duration = input_shape[1] / self.sample_rate
        print(f"Model expects {self.audio_duration} seconds of audio at {self.sample_rate} Hz")
        
        # Load labels
        self.labels = self.load_labels(label_path)
    
    def load_labels(self, label_path):
        """Load class labels from a text file (one label per line)."""
        with open(label_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    
    def preprocess_audio(self, audio, sr, target_sr=48000, duration=3.0):
        """Resample, trim/pad, and format audio to match model input."""
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        
        ################################################################
        #XXXX Temp add preprocessing XXX
        ##################################################################
        # high = 15000 / (target_sr/2)
        # b, a = butter(5, high, btype="low")
        # audio = lfilter(b, a, audio)
        
        high = 15000 / (target_sr/2)
        b, a = butter(5, high, btype="low")
        audio = lfilter(b, a, audio)
        
       
        
        # # Taken from birdnet repository
        # audio = bandpass(audio, 16000, 1, 15000, order=5)

        
        # Calculate required number of samples for the given duration
        required_length = int(target_sr * duration)
        
    
        # Zero padding if the segment is shorter than required, else trim
        if len(audio) < required_length:
            padding = required_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
        else:
            audio = audio[:required_length]
        
        return np.expand_dims(audio.astype(np.float32), axis=0)
    
    def predict_segment(self, audio_segment):
        """Run inference on a single preprocessed audio segment."""
        self.interpreter.set_tensor(self.input_details[0]['index'], audio_segment)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        return predictions
    
    def predict_long_audio(self, audio_path, return_raw_scores=False):
        """ 
        Split a long audio file into fixed-length chunks and classify each segment.
        
        Parameters:
            audio_path (str): Path to the audio file.
            return_raw_scores (bool): If True, return a tuple containing the DataFrame
                with raw logits and confidence scores for all classes (one row per segment)
                along with the final predictions.
        
        Returns:
            - If return_raw_scores is False:
                 A DataFrame with predictions above the confidence threshold.
            - If return_raw_scores is True:
                 A tuple (results_df, raw_scores_df) where:
                     - results_df: DataFrame with predictions above the threshold.
                     - raw_scores_df: DataFrame with all raw scores for all classes, one row per segment.
        """
        y, sr = librosa.load(audio_path, sr=None)  # Load with native sample rate
        segment_length = int(sr * self.audio_duration)  # Samples per segment
        num_segments = int(np.ceil(len(y) / segment_length))
    
        results = []
        raw_scores = []
    
        # 1) Initialize rows list BEFORE looping over segments
        rows = []
        
        for i in range(num_segments):
            start_sample = i * segment_length
            end_sample = min((i + 1) * segment_length, len(y))
            segment = y[start_sample:end_sample]
        
            # Preprocess segment
            processed_segment = self.preprocess_audio(
                segment,
                sr=sr,
                target_sr=self.sample_rate,
                duration=self.audio_duration
            )
        
            # Predict (logit output)
            predictions = self.predict_segment(processed_segment)
        
            # Convert logits to confidence scores using sigmoid
            confidence_scores = scipy.special.expit(predictions)
        
            # 2) Skip this segment entirely if no class exceeds threshold
            if max(confidence_scores) < self.confidence_thresh:
                continue
        
            # 3) Build exactly one row for this segment
            row = {
                "Begin Time (S)": round(start_sample / sr, 2),
                "End Time (S)":   round(end_sample / sr,   2),
                "File":            os.path.basename(audio_path),
                "FilePath":        audio_path,
                "Truth":           Path(audio_path).parts[-2],
            }
        
            # 4) Add one "<label>_score" column per class
            for class_idx, label in enumerate(self.labels):
                confidence_value = confidence_scores[class_idx]
                row[f"{label}"] = round(confidence_value, 4)
        
            # 5) (Optional) If you still want a “winner” column, you can do:
            top_idx = int(np.argmax(confidence_scores))
            row["Predicted Class"] = self.labels[top_idx]
            row["Top Score"]      = round(confidence_scores[top_idx], 4)
        
            # 6) Append this single-row dict to rows
            rows.append(row)
        
            # (Optional) debug print per segment
            #print(f"Segment {i + 1}: kept one row with top label '{self.labels[top_idx]}'")
        
        # 7) After the loop, construct the DataFrame:
        results_df = pd.DataFrame(rows)

        return results_df
    
   
    
    def batch_process_audio_folder(self, 
                                   output_csv="predictions.csv", 
                                   return_raw_scores=False,
                                   thresh=0):
        """
        Recursively process all audio files in a folder and save results to CSV.
        
        Parameters:
            output_csv (str): Path to save the results CSV.
            return_raw_scores (bool): If True, also process and return raw score DataFrames.
        
        Returns:
            - If return_raw_scores is False:
                 A concatenated DataFrame of predictions above the threshold.
            - If return_raw_scores is True:
                 A tuple (final_df, raw_scores_df) where:
                     - final_df: Concatenated DataFrame of predictions above the threshold.
                     - raw_scores_df: Concatenated DataFrame of all raw scores for all classes.
        """
        all_results = []
        all_raw_scores = []  # Store raw scores separately
    
        for root, _, files in os.walk(self.audio_folder):
            for filename in files:
                if filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    audio_path = os.path.join(root, filename)
                    print(f"Processing {audio_path}...")
                    
                    results_df = self.predict_long_audio(audio_path, return_raw_scores=True)
                    all_results.append(results_df)
    
                    # if return_raw_scores:
                    #     results_df, raw_scores_df = self.predict_long_audio(audio_path, return_raw_scores=True)
                    #     all_results.append(results_df)
                    #     all_raw_scores.append(raw_scores_df)
                    # else:
                    #     results_df = self.predict_long_audio(audio_path, return_raw_scores=False)
                    #     all_results.append(results_df)
    
        
        #final_df.to_csv(output_csv, index=False)
        print(f"Batch processing complete! Results saved to {output_csv}")
        
        
        final_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        return final_df
        
        
    
        # if return_raw_scores and all_raw_scores:
        #     raw_scores_df = pd.concat(all_raw_scores, ignore_index=True)
        #     #raw_scores_df.to_csv(output_csv, index=False)
        #     return final_df, raw_scores_df
        # else:
        #     return final_df
    
    def export_to_raven(self, df, raven_file="raven_output.txt"):
        """Export prediction results to a Raven selection table format."""
        df['Selection'] = range(1, df.shape[0] + 1)
        df['Channel'] = 1
        df['View'] = 'Spectrogram 1'
    
        with open(raven_file, 'w') as f:
            # Write header
            f.write("Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tCommon name\tScore\n")
            for _, row in df.iterrows():
                line = f"{row['Selection']}\t{row['View']}\t{row['Channel']}\t{row['Begin Time (S)']}\t{row['End Time (S)']}\t{row['Common name']}\t{row['Score']}\n"
                f.write(line)
    
        print(f"Raven selection table exported to {raven_file}")


class AudioProcessor5:    
    def __init__(self, folder_path=None, segment_duration=2.0, overlap=1.0, 
                  params=None, model=None, detection_thresholds=None,
                  selection_table_name="detections.txt", class_names=None,
                  table_type="selection",outputAllScores = False,
                  retain_detections = True):
        """

        Parameters
        ----------
        folder_path : TYPE, optional
            DESCRIPTION. The default is None.
        segment_duration : TYPE, optional
            DESCRIPTION. The default is 2.0.
        overlap : Float, optional
            DESCRIPTION. Seconds of overlap in the audio advancement.
            The default is 1.0.
        params : TYPE, optional
            DESCRIPTION. The default is None.
        model : TYPE, optional
            DESCRIPTION. The default is None.
        detection_thresholds : TYPE, optional
            DESCRIPTION. The default is None.
        selection_table_name : TYPE, optional
            DESCRIPTION. The default is "detections.txt".
        class_names : TYPE, optional
            DESCRIPTION. The default is None.
        table_type : TYPE, optional
            DESCRIPTION. The default is "selection".

        Returns
        -------
        None.

        """
        self.folder_path = folder_path
        self.audio_files = self.find_audio_files(folder_path) if folder_path else []
        self.segment_duration = segment_duration
        self.overlap = overlap
        self.params = params if params else {
            'outSR': 16000,
            'clipDur': segment_duration,
            'nfft': 512,
            'hop_length': 3200,
            'spec_type': 'mel',
            'rowNorm': True,
            'colNorm': True,
            'rmDCoffset': True,
            'inSR': None
        }
        self.model = model
        self.model_input_shape = model.input_shape[1:] if model else None
        self.detection_thresholds = detection_thresholds if detection_thresholds else {
            class_id: 0.5 for class_id in range(7)
        }
        self.class_names = class_names if class_names else {
            0: 'Abiotic',
            1: 'BKW',
            2: 'HW',
            3: 'NRKW',
            4: 'Offshore',
            5: 'SRKW',
            6: 'Und Bio',
        }
        self.selection_table_name = selection_table_name
        self.table_type = table_type
        self.init_selection_table()
        self.detection_counter = 0
        self._spec_buffer = None  # Buffer for spectrogram optimization
        self.DataSR = 96000
        self.outputAllScores = False
        self.retain_detections = retain_detections
        self.detections = [] if retain_detections else None

    def find_audio_files(self, folder_path):
        return [os.path.join(root, file)
                for root, _, files in os.walk(folder_path)
                for file in files if file.endswith(('.wav', '.mp3', '.flac', '.ogg'))]

    def init_selection_table(self):
        with open(self.selection_table_name, 'w') as f:
            if self.table_type == "selection":
                f.write("Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tLow Freq (Hz)\tHigh Freq (Hz)\tClass\tScore\n")
            elif self.table_type == "sound":
                f.write("Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tLow Freq (Hz)\tHigh Freq (Hz)\tClass\tScore\tSound\n")

    def load_audio_chunk(self, filename, chunk_size):
        with sf.SoundFile(filename) as sf_file:
            self.DataSR = sf_file.samplerate
            self.params['inSR'] = self.DataSR
            while True:
                y = sf_file.read(frames=chunk_size, dtype='float32', always_2d=False)
                if len(y) == 0:
                    break
                yield y, self.DataSR

    def create_segments_streaming(self, y, sr):
        segment_length = int(self.segment_duration * sr)
        overlap_length = int(self.overlap * sr)
        
        start = 0
        while start + segment_length <= len(y):
            yield y[start:start + segment_length], start / sr
            start += segment_length - overlap_length

    def create_spectrogram(self, y):
        if self._spec_buffer is None:  # Check if buffer is None (instead of directly checking the array)
            self._spec_buffer = np.zeros((self.params['nfft'] // 2 + 1, self.model_input_shape[1]), dtype=np.float32)
        
        spectrogram = create_spectrogram(y, return_snr=False, **self.params)
        expected_time_steps = self.model_input_shape[1]
        
        if spectrogram.shape[1] < expected_time_steps:
            self._spec_buffer[:, :spectrogram.shape[1]] = spectrogram
            return self._spec_buffer
        return spectrogram[:, :expected_time_steps]

    def process_batch(self, batch_segments, batch_start_times, batch_files):
        batch_segments = np.stack(batch_segments)  # Create a batch array
        predictions = self.model.predict(batch_segments)  # Model predictions in a batch

        # Two options for output, kick out all the scores (untraditional) or 
        # kick out the scores for the argmax, more traditional
        if self.outputAllScores == False:
            
            # Convert to a numerical array
            numerical_predictions = np.round(np.array(predictions.tolist()), 3)
            
            for i, row in enumerate(numerical_predictions):
                row = np.array(row, dtype=float)  # Convert the row to a numerical array
                max_class = np.argmax(row)
                max_score = np.max(row)
                startTime = batch_start_times[i]
                stopTime = batch_start_times[i] + self.segment_duration
                
                if max_score >= self.detection_thresholds[max_class]:
                    self.output_detection(
                        class_id =max_class, 
                        score= max_score,   
                        start_time= startTime, 
                        end_time= stopTime,
                        filename =batch_files[i]
                    )
                    
        else: # write out all of the scores above the detection threshold
            for i, prediction in enumerate(predictions):
                for class_id, score in enumerate(prediction):
                    if score >= self.detection_thresholds[class_id]:
                        self.output_detection(
                            class_id, score, 
                            batch_start_times[i], 
                            batch_start_times[i] + self.segment_duration,
                            batch_files[i]
                        )
             
            

    def output_detection(self, class_id, score, start_time, end_time, filename):
        
        selection = self.detection_counter
        class_name = self.class_names[class_id]
        
        self.detection_counter += 1
        if self.retain_detections:
               detection = {
                   "Selection": self.detection_counter + 1,
                   "View": "Spectrogram",
                   "Channel": 1,
                   "Begin Time (S)": start_time,
                   "End Time (S)": end_time,
                   "Low Freq (Hz)": 0,
                   "High Freq (Hz)": 8000,
                   "Class": class_name,
                   "Score": score,
                   "File": filename}
               self.detections.append(detection)
        
        with open(self.selection_table_name, 'a') as f:
            if self.table_type == "selection":
                f.write(f"{selection}\tSpectrogram\t1\t{start_time:.6f}\t{end_time:.6f}\t0\t8000\t{class_name}\t{score:.4f}\n")
            elif self.table_type == "sound":
                f.write(f"{selection}\tSpectrogram\t1\t{start_time:.6f}\t{end_time:.6f}\t0\t8000\t{class_name}\t{score:.4f}\t{filename}\n")

    def get_detections(self, as_dataframe=True):
        '''
        # Returns a pandas dataframe of the Raven selection table

        Parameters
        ----------
        as_dataframe : TYPE, optional
            DESCRIPTION. The default is True.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if not self.retain_detections:
            raise ValueError("Retention of detections was not enabled. Set retain_detections=True in the constructor.")
        return pd.DataFrame(self.detections) if as_dataframe else np.array(self.detections)
    
    def process_all_files(self):
        '''
        

        Returns
        -------
        None.

        '''
        filestreamStart = 0  # Initialize the global start time
    
        for filename in self.audio_files:
            print(f"Processing file: {filename}")
            chunk_start_time = 0  # Initialize chunk start time for each file
            batch_segments, batch_start_times, batch_files = [], [], []
    
            # Process each audio chunk
            for audio_chunk, sr in self.load_audio_chunk(filename, chunk_size=self.DataSR * 15):  # Process 15-second chunks
                for segment, start_time in self.create_segments_streaming(audio_chunk, self.DataSR):
                    # Adjust the segment's start time to be relative to the whole filestream
                    global_start_time = filestreamStart + chunk_start_time + start_time
                    batch_segments.append(self.create_spectrogram(segment))
                    batch_start_times.append(global_start_time)
                    batch_files.append(filename)
    
                    # If batch size is reached, process the batch
                    if len(batch_segments) == 32:
                        self.process_batch(batch_segments, batch_start_times, batch_files)
                        batch_segments, batch_start_times, batch_files = [], [], []
    
                # Update chunk_start_time for the next chunk
                chunk_start_time += len(audio_chunk) / sr
    
            # Process remaining segments if any
            if batch_segments:
                self.process_batch(batch_segments, batch_start_times, batch_files)
    
            # Update the global filestreamStart after processing the current file
            filestreamStart += chunk_start_time  # Increment global filestream start time


import os
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import scipy.special
from scipy.signal import butter, lfilter
import tensorflow.lite as tflite
import time
class BirdNetPredictorNew:
    def __init__(self, model_path, label_path, audio_folder, 
                 sample_rate=48000, 
                 audio_duration=3.0, 
                 confidence_thresh=0.5):
        self.model_path = model_path
        self.label_path = label_path
        self.audio_folder = audio_folder
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.confidence_thresh = confidence_thresh

        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Use input shape to infer duration
        input_shape = self.input_details[0]['shape']
        self.audio_duration = input_shape[1] / self.sample_rate
        print(f"Model expects {self.audio_duration} seconds of audio at {self.sample_rate} Hz")

        self.labels = self.load_labels(label_path)

    def load_labels(self, label_path):
        with open(label_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def preprocess_audio(self, audio, sr, target_sr=48000, duration=3.0):
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # high = 16000 / (target_sr / 2)
        # b, a = butter(5, high, btype="low")
        # audio = lfilter(b, a, audio)

        required_length = int(target_sr * duration)
        if len(audio) < required_length:
            audio = np.pad(audio, (0, required_length - len(audio)), mode='constant')
        else:
            audio = audio[:required_length]

        return np.expand_dims(audio.astype(np.float32), axis=0)

    def predict_batch(self, audio_batch):
        """
        Run inference on a batch of preprocessed segments. Expects shape (batch_size, samples).
        """
        self.interpreter.resize_tensor_input(self.input_details[0]['index'], audio_batch.shape)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_details[0]['index'], audio_batch)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
        return predictions

    def predict_long_audio(self, audio_path, return_raw_scores=False):
        y, sr = librosa.load(audio_path, sr=None)
        segment_length = int(sr * self.audio_duration)
        num_segments = int(np.ceil(len(y) / segment_length))

        processed_segments = []
        start_ends = []

        for i in range(num_segments):
            start_sample = i * segment_length
            end_sample = min((i + 1) * segment_length, len(y))
            segment = y[start_sample:end_sample]
            processed = self.preprocess_audio(segment, sr=sr, target_sr=self.sample_rate, duration=self.audio_duration)
            processed_segments.append(processed)
            start_ends.append((start_sample, end_sample))

        if not processed_segments:
            return pd.DataFrame()

        audio_batch = np.vstack(processed_segments)
        predictions = self.predict_batch(audio_batch)

        rows = []
        for i, prediction in enumerate(predictions):
            confidence_scores = scipy.special.expit(prediction)

            if max(confidence_scores) < self.confidence_thresh:
                continue

            start_sample, end_sample = start_ends[i]

            row = {
                "Begin Time (S)": round(start_sample / sr, 2),
                "End Time (S)": round(end_sample / sr, 2),
                "File": os.path.basename(audio_path),
                "FilePath": audio_path,
                "Truth": Path(audio_path).parts[-2],
            }

            for class_idx, label in enumerate(self.labels):
                row[f"{label}"] = round(confidence_scores[class_idx], 4)

            top_idx = int(np.argmax(confidence_scores))
            row["Predicted Class"] = self.labels[top_idx]
            row["Top Score"] = round(confidence_scores[top_idx], 4)

            rows.append(row)

        results_df = pd.DataFrame(rows)
        return results_df
    def batch_process_audio_folder(self, 
                               output_csv="predictions.csv", 
                               return_raw_scores=False,
                               batch_size=64):
        """
        Process all audio files in the folder in batches of N files.
        """
        all_results = []
        batch_audio = []
        batch_paths = []
    
        for root, _, files in os.walk(self.audio_folder):
            audio_files = [
                os.path.join(root, f)
                for f in files
                if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))
            ]
            print(f"Found {len(audio_files)} audio files...")
           
            nbatches = np.ceil(len(audio_files)/batch_size)
            
            
            # do stuff
            
            for j, audio_path in enumerate(audio_files):

                # t = time.time()
                y, sr = librosa.load(audio_path, sr=None)
                processed = self.preprocess_audio(
                    y, sr=sr, target_sr=self.sample_rate, duration=self.audio_duration
                )
                batch_audio.append(processed)
                batch_paths.append(audio_path)
    
                # When we hit the batch size, run prediction
                if len(batch_audio) == batch_size:
                    batch_array = np.vstack(batch_audio)
                    predictions = self.predict_batch(batch_array)
    
                    for i, prediction in enumerate(predictions):
                        confidence_scores = scipy.special.expit(prediction)
                        if max(confidence_scores) < self.confidence_thresh:
                            continue
    
                        row = {
                            "Begin Time (S)": 0,
                            "End Time (S)": round(self.audio_duration, 2),
                            "File": os.path.basename(batch_paths[i]),
                            "FilePath": batch_paths[i],
                            "Truth": Path(batch_paths[i]).parts[-2],
                        }
    
                        for class_idx, label in enumerate(self.labels):
                            row[f"{label}"] = round(confidence_scores[class_idx], 4)
    
                        top_idx = int(np.argmax(confidence_scores))
                        row["Predicted Class"] = self.labels[top_idx]
                        row["Top Score"] = round(confidence_scores[top_idx], 4)
    
                        all_results.append(row)
                        
    
                    # Clear the batch
                    batch_audio = []
                    batch_paths = []
                    # elapsed = time.time() - t
                    # elapsed/batch_size
                    
                    print(f' {j} of {len(audio_files)} files done')
    
                        
            # Final partial batch (if any)
            if batch_audio:
                batch_array = np.vstack(batch_audio)
                predictions = self.predict_batch(batch_array)
    
                for i, prediction in enumerate(predictions):
                    confidence_scores = scipy.special.expit(prediction)
                    if max(confidence_scores) < self.confidence_thresh:
                        continue
    
                    row = {
                        "Begin Time (S)": 0,
                        "End Time (S)": round(self.audio_duration, 2),
                        "File": os.path.basename(batch_paths[i]),
                        "FilePath": batch_paths[i],
                        "Truth": Path(batch_paths[i]).parts[-2],
                    }
    
                    for class_idx, label in enumerate(self.labels):
                        row[f"{label}"] = round(confidence_scores[class_idx], 4)
    
                    top_idx = int(np.argmax(confidence_scores))
                    row["Predicted Class"] = self.labels[top_idx]
                    row["Top Score"] = round(confidence_scores[top_idx], 4)
    
                    all_results.append(row)
    
        final_df = pd.DataFrame(all_results)
        #elapsed = time.time() - t
        print(f"Batch processing complete! {len(final_df)} predictions.")
        
        return final_df


    def export_to_raven(self, df, raven_file="raven_output.txt"):
        df['Selection'] = range(1, df.shape[0] + 1)
        df['Channel'] = 1
        df['View'] = 'Spectrogram 1'

        with open(raven_file, 'w') as f:
            f.write("Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tCommon name\tScore\n")
            for _, row in df.iterrows():
                f.write(f"{row['Selection']}\t{row['View']}\t{row['Channel']}\t{row['Begin Time (S)']}\t"
                        f"{row['End Time (S)']}\t{row['Predicted Class']}\t{row['Top Score']}\n")

        print(f"Raven selection table exported to {raven_file}")



import os
import numpy as np
import soundfile as sf
import tensorflow as tf  # Ensure TensorFlow is imported for TFLite compatibility

# class BirdnetAudioProcessor5:
#     def __init__(self, folder_path=None, segment_duration=2.0, overlap=1.0,
#                  model_path=None, label_path=None, detection_thresholds=None,
#                  selection_table_name="detections.txt", class_names=None,
#                  table_type="selection", outputAllScores=False, retain_detections=True):
#         self.folder_path = folder_path
#         self.audio_files = self.find_audio_files(folder_path) if folder_path else []
#         self.segment_duration = segment_duration
#         self.overlap = overlap
#         self.model_path = model_path
#         self.model = self.load_tflite_model(model_path)
#         self.labels = self.load_labels(label_path)
#         self.detection_thresholds = detection_thresholds if detection_thresholds else {i: 0.5 for i in range(len(self.labels))}
#         self.class_names = class_names if class_names else self.labels
#         self.selection_table_name = selection_table_name
#         self.table_type = table_type
#         self.init_selection_table()
#         self.detection_counter = 0
#         self.outputAllScores = outputAllScores
#         self.retain_detections = retain_detections
#         self.detections = [] if retain_detections else None

#     def load_tflite_model(self, model_path):
#         # Load the TensorFlow Lite model
#         interpreter = tf.lite.Interpreter(model_path=model_path)
#         interpreter.allocate_tensors()
#         return interpreter

#     def load_labels(self, label_path):
#         with open(label_path, 'r') as file:
#             labels = [line.strip() for line in file.readlines()]
#         return labels

#     def find_audio_files(self, folder_path):
#         return [os.path.join(root, file)
#                 for root, _, files in os.walk(folder_path)
#                 for file in files if file.endswith(('.wav', '.mp3', '.flac', '.ogg'))]

#     def init_selection_table(self):
#         with open(self.selection_table_name, 'w') as f:
#             f.write("Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tLow Freq (Hz)\tHigh Freq (Hz)\tClass\tScore\n")

#     def load_audio_chunk(self, filename, chunk_size):
#         with sf.SoundFile(filename) as sf_file:
#             while True:
#                 y = sf_file.read(frames=chunk_size, dtype='float32', always_2d=False)
#                 if len(y) == 0:
#                     break
#                 yield y, sf_file.samplerate

#     def create_segments_streaming(self, y, sr):
#         segment_length = int(self.segment_duration * sr)
#         overlap_length = int(self.overlap * sr)
#         start = 0
#         while start + segment_length <= len(y):
#             yield y[start:start + segment_length], start / sr
#             start += segment_length - overlap_length

#     def process_batch(self, batch_segments, batch_start_times, batch_files):
#         input_details = self.model.get_input_details()
#         output_details = self.model.get_output_details()
#         for i, segment in enumerate(batch_segments):
#             # Prepare the input data format for TFLite (typically this involves normalizing and resizing)
#             segment = np.resize(segment, input_details[0]['shape'])
#             self.model.set_tensor(input_details[0]['index'], segment.astype('float32'))
#             self.model.invoke()
#             predictions = self.model.get_tensor(output_details[0]['index'])[0]

#             max_class_id = np.argmax(predictions)
#             max_score = predictions[max_class_id]

#             if max_score >= self.detection_thresholds[max_class_id]:
#                 self.output_detection(
#                     class_id=max_class_id,
#                     score=max_score,
#                     start_time=batch_start_times[i],
#                     end_time=batch_start_times[i] + self.segment_duration,
#                     filename=batch_files[i]
#                 )

#     def output_detection(self, class_id, score, start_time, end_time, filename):
#         selection = self.detection_counter
#         class_name = self.class_names[class_id]
#         self.detection_counter += 1
#         with open(self.selection_table_name, 'a') as f:
#             f.write(f"{selection}\tSpectrogram\t1\t{start_time:.6f}\t{end_time:.6f}\t0\t8000\t{class_name}\t{score:.4f}\n")

#     def process_all_files(self):
#         for filename in self.audio_files:
#             print(f"Processing file: {filename}")
#             for audio_chunk, sr in self.load_audio_chunk(filename, chunk_size=96000 * 15):  # Assuming 15 seconds chunks
#                 for segment, start_time in self.create_segments_streaming(audio_chunk, sr):
#                     self.process_batch([segment], [start_time], [filename])

# # Example usage
# processor = AudioProcessor5(
#     folder_path="path/to/audio_folder",
#     model_path="path/to/birdnet_model.tflite",
#     label_path="path/to/labels.txt",
#     detection_thresholds={0: 0.8, 1: 0.7, 2: 0.6},  # Custom detection thresholds per class if needed
#     class_names=["Abiotic", "BKW", "HW", "NRKW", "Offshore", "SRKW", "Und Bio"]
# )
# processor.process_all_files()





 