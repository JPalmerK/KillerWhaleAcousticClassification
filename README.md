# Ecotype Final
This repository contains the the metadata and outputs associated with the publication **Population-level acoustic classification of Salish Sea killer whales: integrating biologically informed call-type balancing to build robust models for conservation monitoring**. 


## BirdNET Models
Pre-trained [BirdNET](https://github.com/birdnet-team/BirdNET-Analyzer) models are available in the BirdNET 01-09 folders as are other files needed to run the models using the BirdNet analyzer. Compressed training data in the form of NPZ files are similarly available for training, retraining, or critiquing existing models. 

## Scripts
**ExportAllClips.py** - script used export 3 second audio clips from all csv files for each BirdNet model

**EcotypeDefs.py** - script containing python definitions to load audio files, tflite models and predict on single or batch audio data.

**BirdnetEval_organized_birdnetGrid.py** - script for running models on evaulation data (requires EcotypeDefs.py) and creating evaluation plots.

**CreateExperimentalData.R** Code to compile the [DCLDE data and annotations](https://www.nature.com/articles/s41597-025-05281-5) into structured classes for model training.






