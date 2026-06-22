Cardiac Phase Detection in Multi-View Multi-Disease Cardiac Magnetic Resonance Images 
==============================

This repository contains code to perform automatic phase detection from short-axis (SAX) and four-chamber long-axis (4CH)
cardiac magnetic resonance (CMR) cine images.

This repository was used for the following paper:

**Deformable image registration for self-supervised cardiac phase detection in cardiac magnetic resonance images of patients with various diseases**

For a more detailed description, a model definition and further results we refer to our 
<a target="_blank" href="https://www.sciencedirect.com/science/article/pii/S1361841526002112?via%3Dihub">paper</a>.



Motivation
-


![Visual Abstract of Pipeline](/docs/img/Visual%20Abstract.png)


Abstract
=
Cardiovascular magnetic resonance (CMR) is widely used to assess cardiac function, but individual cardiac cycles complicate automatic temporal comparison and sub-phase analysis. Accurate cardiac keyframe detection can eliminate this problem. However, automatic methods solely derive end-systole (ES) and end-diastole (ED) frames from left ventricular volume curves, which do not provide a deeper insight into myocardial motion.

We propose a self-supervised deep learning method detecting five keyframes in short-axis (SAX) and four-chamber (4CH) cine CMR. Initially, dense deformable registration fields are derived from CMR to compute a 1D motion descriptor encoding global cardiac contraction and relaxation patterns. Keyframes are derived from these characteristic curves with a set of rules.
The method was independently evaluated for both views using four databases encompassing multiple centre, vendor and disease. M&Ms-2 (n=360) was used for training and evaluation; M&Ms (n=345) and ACDC (n=100) for repeatability control. Generalisability to patients with rare congenital heart defects was tested using the German Competence Network (GCN) database. A disease-stratified analysis confirmed stable performance across cardiomyopathies and congenital abnormalities.

Our method improved detection accuracy by 49%/59% for SAX and 31%/39% for 4CH in ED/ES over the volume-based approach, with mean cyclic frame difference (cFD) below 1.3 and 1.2 frames for SAX and 4CH respectively. Our framework enables temporally aligned inter- and intra-patient analysis of cardiac dynamics, irrespective of cycle or phase lengths for aligned strain analysis or temporal normalisation. 

How to get started?
------------
1. **Environment setup**

    Follow the <a target="_blank" href="https://github.com/Cardio-AI/cmr-multi-view-phase-detection/tree/main/docs/Setup.md">Setup Guide</a> to install dependencies and configure your environment.

2. **Data Preparation** 

    See the <a target="_blank" href="https://github.com/Cardio-AI/cmr-multi-view-phase-detection/tree/main/docs/Data.md">Data Guide</a> for instructions on how to structure your dataset.

3. **Run Training and Inference**

    - You can start directly from the example notebooks in the ``notebooks`` folder
    - Alternatively, you can run the scripts from your preferred IDE or command line.

Follow the links below for more details on:
- <a target="_blank" href="https://github.com/Cardio-AI/cmr-multi-view-phase-detection/tree/main/docs/Training.md">Training</a>
- <a target="_blank" href="https://github.com/Cardio-AI/cmr-multi-view-phase-detection/tree/main/docs/Keyframe-detection.md">Keyframe detection</a>
- <a target="_blank" href="https://github.com/Cardio-AI/cmr-multi-view-phase-detection/tree/main/docs/Phase2Phase-LAS.md">Phase to phase LAS computing</a>

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like 'make environment' or 'make requirement'
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── metadata       <- Excel and csv files with additional metadata
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── predicted      <- Model predictions, will be used for the evaluations
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │   ├── Dataset        <- call the dataset helper functions, analyze the datasets
    │   ├── Evaluate       <- Evaluate the model performance, create plots
    │   ├── Predict        <- Use the models on new data
    │   ├── Train          <- Train a new model
    │   └── Test_IO        <- IO tests
    │   └── Test_Models    <- Tensorflow functional or subclassing tests
    │
    ├── exp            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── configs        <- Experiment config files as json
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    │   ├── history        <- Tensorboard trainings history files
    │   ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   └── tensorboard_logs  <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Helper functions that will be used by the notebooks.
        ├── data           <- create, preprocess and extract the nrrd files
        ├── models         <- Modelzoo, Modelutils and Tensorflow layers
        ├── utils          <- Metrics, callbacks, io-utils, notebook imports
        └── visualization  <- Plots for the data, generator or evaluations


Paper:
--------
Please cite the following paper if you use/modify or adapt parts of this repository:

**Bibtext**
```
@article{MUELLER_KOEHLER_2026104142,
title = {Deformable image registration for self-supervised cardiac phase detection in cardiac magnetic resonance images of patients with various diseases},
journal = {Medical Image Analysis},
pages = {104142},
year = {2026},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2026.104142},
url = {https://www.sciencedirect.com/science/article/pii/S1361841526002112},
author = {Sarah Kaye Mueller and Sven Koehler and Jonathan Kiekenap and Gerald Greil and Tarique Hussain and Samir Sarikouch and Florian Andre and Norbert Frey and Sandy Engelhardt},
keywords = {Cardiac phase detection, Cardiac motion description, Cardiac magnetic resonance imaging, Self-supervised learning, Discrete vector fields},
abstract = {Cardiovascular magnetic resonance (CMR) is widely used to assess cardiac function, but individual cardiac cycles complicate automatic temporal comparison and sub-phase analysis. Accurate cardiac keyframe detection can eliminate this problem. However, automatic methods solely derive end-systole (ES) and end-diastole (ED) frames from left ventricular volume curves, which do not provide a deeper insight into myocardial motion. We propose a self-supervised deep learning method detecting five keyframes in short-axis (SAX) and four-chamber (4CH) cine CMR. Initially, dense deformable registration fields are derived from CMR to compute a 1D motion descriptor encoding global cardiac contraction and relaxation patterns. Keyframes are derived from these characteristic curves with a set of rules. The method was independently evaluated for both views using four databases encompassing multiple centre, vendor and disease. M&Ms-2 (n=360) was used for training and evaluation; M&Ms (n=345) and ACDC (n=100) for repeatability control. Generalisability to patients with rare congenital heart defects was tested using the German Competence Network (GCN) database. A disease-stratified analysis confirmed stable performance across cardiomyopathies and congenital abnormalities. Our method improved detection accuracy by 49%/59% for SAX and 31%/39% for 4CH in ED/ES over the volume-based approach, with mean cyclic frame difference (cFD) below 1.3 and 1.2 frames for SAX and 4CH respectively. Our framework enables temporally aligned inter- and intra-patient analysis of cardiac dynamics, irrespective of cycle or phase lengths for aligned strain analysis or temporal normalisation. Code and annotations are available at: https://github.com/Cardio-AI/cmr-multi-view-phase-detection.git.}
}

```


Affiliation
--------
For more Information of our work, please visit our website:
<a target="_blank" href="https://www.klinikum.uni-heidelberg.de/chirurgische-klinik-zentrum/herzchirurgie/forschung/institute-for-artificial-intelligence-in-cardiovascular-medicine-aicm">Institute for Artificial Intelligence in Cardiovascular Medicine </a>

We are part of the Department of Cardiology, Angiology, Pneumology, Heidelberg University Hospital, Heidelberg, Germany
