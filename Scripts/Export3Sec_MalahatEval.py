# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:18:42 2024

@author: kaity
"""
import numpy as np
import pandas as pd
import librosa
import os
import soundfile as sf
# Create data for Birdnet
annotations = pd.read_csv("C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\Malahat_Holdout_parent_redo.csv")


# Check the lables look good
annotations.Labels.value_counts()


# Spectrogram parameters
params = {
    'clipDur': 3,
    'outSR': 48000,
    'fmin': 0
}



# Iterate over each annotation
for idx, row in annotations.iloc[0:].iterrows():
    file_path = row['FilePath']
    start_time = row['FileBeginSec']
    end_time = row['FileEndSec']
    dep = row['Dataset']
    provider = row['Provider']
    utc = row['UTC']
    speciesFolder = row['Labels']
    SoundFilee = row['Soundfile']
    
    # Define the base output directory and the species-specific folder
    base_output_dir = "C:\\TempData\\Malahat_HOLDOUT\\"
    output_folder = os.path.join(base_output_dir, speciesFolder)
    
    # Create the species folder if it doesn't exist
    if os.path.exists(output_folder)==False:
        os.makedirs(output_folder, exist_ok=True)
    
    # Load and process the audio segment
    file_duration = librosa.get_duration(path=file_path)
    duration = end_time - start_time
    center_time = start_time + (duration / 2.0)
    new_start_time = center_time -( params['clipDur'] / 2)
    new_end_time = center_time + (params['clipDur'] / 2)
    
    if new_end_time - new_start_time < params['clipDur']:
        pad_length = params['clipDur'] - (new_end_time - new_start_time)
        new_start_time = max(0, new_start_time - pad_length / 2.0)
        new_end_time = min(file_duration, new_end_time + pad_length / 2.0)
    
    new_start_time = max(0, min(new_start_time, file_duration - params['clipDur']))
    new_end_time = max(params['clipDur'], min(new_end_time, file_duration))
    
    
    
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
    
    # Scale between 0 and 1 
    audio_data = (audio_data - np.min(audio_data)) / (np.max(audio_data) - np.min(audio_data))
    # Define the output file path
    clipName = f"{SoundFilee}_{provider}_{dep}_{str(idx)}.wav"
    output_file_path = os.path.join(output_folder, clipName)
    
    # Export the audio clip
    sf.write(output_file_path, audio_data, sample_rate)
    print(f"Saved {output_file_path}")
