# Killer Whale Acoustic Classification
[![DOI](https://zenodo.org/badge/1102105124.svg)](https://doi.org/10.5281/zenodo.18209486)

This repository contains the the metadata and outputs associated with the publication **Population-level acoustic classification of Salish Sea killer whales: integrating biologically informed call-type balancing to build robust models for conservation monitoring**. 



## 📁 Repository Structure

```text
/
├── BirdNETmODELS/         # Head directory containing all trained birdNET models and scripts
├──   BirdNET_01–09/       # Pre-trained BirdNET models and required analyzer files
├── scripts/               # Python and R scripts for data prep and evaluation
└── results/               # Selected evaluation outputs and figures (if included)
```
The BirdNET_01–09 folders contain pre-trained models formatted for easy use with BirdNET.
Each folder includes:

  .tflite model file
  .npz compressed training data file used during model development; these are provided for transparency and optional advanced use but are not required for running predictions.
  Corresponding class/label mapping file
  Model training parameters
  Screenshot of learning training history from Birdnet API

## BirdNET Models
Pre-trained [BirdNET](https://github.com/birdnet-team/BirdNET-Analyzer) models are available in the BirdNET 01-09 folders as are other files needed to run the models using the BirdNet analyzer. Compressed training data in the form of NPZ files are similarly available for training, retraining, or critiquing existing models. 

## Scripts
**ExportAllClips.py** - script used export 3 second audio clips from all csv files for each BirdNet model
**EcotypeDefs.py** - script containing python definitions to load audio files, tflite models and predict on single or batch audio data.
**BirdnetEval_organized_birdnetGrid.py** - script for running models on evaulation data (requires EcotypeDefs.py) and creating evaluation plots.
**CreateExperimentalData.R** Code to compile the [DCLDE data and annotations](https://www.nature.com/articles/s41597-025-05281-5) into structured classes for model training.
**

Example of loading trained tensor flow lite model and creating raven selection table
```
  # Create a prediction object pointing to tflite model, labels from BirdNET, and audio folder. Default classificaiton threshold is 0.9, change individual classes using a dictionary
    pred = BirdNetPredictorNew(
        model_path=r"C:\Users\kaity\Documents\GitHub\EcotypeFinal\BirdNET Models\birdnet07\birdnet07_8khz_cutoff.tflite",
        label_path=r"C:\Users\kaity\Documents\GitHub\EcotypeFinal\BirdNET Models\birdnet07\birdnet07_8khz_cutoff_Labels.txt",
        audio_folder=r"E:\AdriftData\Adrift_040",
        confidence_thresh=0.6,                 # default for all classes
        class_thresholds={"SRKW": 0.95, 'TKW': 0.90, 'HW': 0.9},        # stricter threshold just for SRKW
        recursive=True,
    )

    # Stream model predictions to Raven Selection Table
    pred.predict_folder_global_raven_streaming_to_file(
        raven_file=r"C:\TempData\Adrift_040_streaming.txt",
        hop_s=1.5,
        channels=1,           # int index of channel to pick (0-based). Raven Channel will be 1 (mixed/picked), indexing starts at 0
        batch_size=16,
        recursive=True,
        low_hz=0,
        high_hz=24000,
        include_file_cols=False,
        view="Spectrogram 1",
    )


```
## Predictor Options
Channel selection options. Select one of the following for multi-channel data
* "mix" (default): average all channels -> single timeline, Raven Channel=1
* int (e.g., 0): pick one channel -> Raven Channel=1
* list (e.g., [0,1]): average those channels -> Raven Channel=1
* "all": run each channel separately -> Raven Channel = actual channel index + 1






