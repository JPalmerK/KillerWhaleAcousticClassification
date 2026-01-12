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

Exampel of loading trained tensor flow lite model and creating raven selection table
```
    pred = BirdNetPredictorNew(
        model_path="C:/Users/kaity/Documents/GitHub/EcotypeFinal/BirdNET Models/birdnet07/birdnet07.tflite",
        label_path="C:/Users/kaity/Documents/GitHub/EcotypeFinal/BirdNET Models/birdnet07/birdnet07_8khz_cutoff_Labels.txt",
        audio_folder="C:\\TempData\\TestDays\\Biggs",
        confidence_thresh=0.9,
    )
    
    df = pred.predict_folder_global_raven(hop_s=1.5, recursive=True)
    pred.export_to_raven(df, "C:/TempData\\malahat_global_raven.txt")
```






