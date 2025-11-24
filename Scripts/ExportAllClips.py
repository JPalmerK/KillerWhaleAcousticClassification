# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 13:36:26 2025

@author: kaity
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

# Spectrogram / audio clip parameters
params = {
    'clipDur': 3,
    'outSR': 48000,
    'fmin': 0
}

# Root folders
csv_root   = r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\BirdnetGrid\birdnet02\\"
output_root = r"C:\TempData\TKWCalls"

# Iterate over all CSVs in csv_root (no BirdnetXX subfolders)
for fname in sorted(os.listdir(csv_root)):
    if not fname.lower().endswith(".csv"):
        continue

    csv_path = os.path.join(csv_root, fname)
    exp_name = os.path.splitext(fname)[0]  # e.g., Grid_SRKW30_TKW10

    print(f"🟢 Processing {csv_path}")
    annotations = pd.read_csv(csv_path)

    # Output path: <output_root>/<CSV_BASENAME>/
    base_output_dir = os.path.join(output_root, exp_name)
    os.makedirs(base_output_dir, exist_ok=True)

    # -------------- keep your existing row loop below unchanged --------------
    for idx, row in annotations.iterrows():
        try:
            file_path = row['FilePath']
            start_time = row['FileBeginSec']
            end_time = row['FileEndSec']
            dep = row['Dataset']
            provider = row['Provider']
            speciesFolder = row['Labels']
            soundfile = row['Soundfile']
            CT = row['CalltypeCategory']

            output_folder = os.path.join(base_output_dir, speciesFolder)
            os.makedirs(output_folder, exist_ok=True)

            file_duration = librosa.get_duration(path=file_path)
            duration = end_time - start_time
            center_time = start_time + duration / 2.0
            new_start_time = max(0, center_time - params['clipDur'] / 2.0)
            new_end_time = new_start_time + params['clipDur']
            if new_end_time > file_duration:
                new_start_time = max(0, file_duration - params['clipDur'])
                new_end_time = file_duration

            audio_data, sample_rate = librosa.load(
                file_path,
                #sr=params['outSR'],
                offset=new_start_time,
                duration=params['clipDur'],
                mono=False
            )
            if audio_data.ndim > 1:
                print(f"⚠️  Multi-channel audio ({audio_data.shape[0]} channels); retaining first")
                audio_data = audio_data[0]

            if np.max(audio_data) != np.min(audio_data):
                audio_data = (audio_data - np.min(audio_data)) / (np.max(audio_data) - np.min(audio_data))

            clip_name = f"{CT}_{provider}_{dep}_{str(idx)}.wav"
            output_path = os.path.join(output_folder, clip_name)

            sf.write(output_path, audio_data, sample_rate)
            print(f"✅ Saved: {output_path}")

        except Exception as e:
            print(f"❌ Failed on row {idx}: {e}")
