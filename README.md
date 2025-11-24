# Acoustic Classification
This repository contains the the metadata and outputs associated with the publication **Population-level acoustic classification of Salish Sea killer whales: integrating biologically informed call-type balancing to build robust models for conservation monitoring**. 

## 📁 Repository Structure

```text
/
├── BirdNETmODELS/         # Head directory containing all trained birdNET models and scripts
├──   BirdNET_01–09/       # Pre-trained BirdNET models and required analyzer files
├── scripts/               # Python and R scripts for data prep and evaluation
└── results/               # Selected evaluation outputs and figures (if included)
```

## BirdNET Models
Pre-trained [BirdNET](https://github.com/birdnet-team/BirdNET-Analyzer) models are available in the BirdNET 01-09 folders as are other files needed to run the models using the BirdNet analyzer. Compressed training data in the form of NPZ files are similarly available for training, retraining, or critiquing existing models. 

## Scripts
**ExportAllClips.py** - script used export 3 second audio clips from all csv files for each BirdNet model
**EcotypeDefs.py** - script containing python definitions to load audio files, tflite models and predict on single or batch audio data.

**BirdnetEval_organized_birdnetGrid.py** - script for running models on evaulation data (requires EcotypeDefs.py) and creating evaluation plots.

**CreateExperimentalData.R** Code to compile the [DCLDE data and annotations](https://www.nature.com/articles/s41597-025-05281-5) into structured classes for model training.


The BirdNET_01–09 folders contain pre-trained models formatted for easy use with BirdNET.
Each folder includes:

.tflite model file

Corresponding class/label mapping file

Any additional configuration required by BirdNET Analyzer

These files can be dropped directly into BirdNET Analyzer or referenced in BirdNET-API to classify:

Southern Resident killer whales (SRKW)

West Coast Transients (TKW)

Humpback whales (HW)

Background noise (BG)

The training_npz/ directory contains compressed training data used during model development; these are provided for transparency and optional advanced use but are not required for running predictions.



