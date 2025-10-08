Cardiac Phase Detection in Multi-View Multi-Disease Cardiac Magnetic Resonance Images 
==============================

This repository contains code to perform automatic phase detection from short-axis (SAX) and four-chamber long-axis (4CH)
cardiac magnetic resonance (CMR) cine images.

This repository was used for the following paper:

**Deformable Image Registration for Self-supervised Cardiac Phase Detection in Multi-View Multi-Disease Cardiac Magnetic Resonance Images**

For a more detailed description, a model definition and further results we refer to our 
<a target="_blank" href="https://link.springer.com/chapter/10.1007/978-3-031-94562-5_11">paper</a>.



Motivation
-


![Visual Abstract of Pipeline](/docs/img/Visual%20Abstract.png)


Abstract
=
Cardiovascular magnetic resonance (CMR) is the gold standard for assessing cardiac function, but individual cardiac
cycles complicate automatic temporal comparison or sub-phase analysis. Accurate cardiac keyframe detection 
can eliminate this problem. However, automatic methods solely derive end-systole (ES) and end-diastole (ED) 
frames from left ventricular volume curves, which do not provide a deeper insight into myocardial motion.

We propose a self-supervised deep learning method detecting five keyframes in short-axis (SAX) and four-chamber 
long-axis (4CH) cine CMR. Initially, dense deformable registration fields are derived from the images and used 
to compute a 1D motion descriptor, which provides valuable insights into global cardiac contraction and 
relaxation patterns. From these characteristic curves, keyframes are determined using a simple set of rules.

The method was independently evaluated for both views using three public, multicentre, multidisease datasets. 
M&Ms-2 (n=360) dataset was used for training and evaluation, and M&Ms (n=345) and ACDC (n=100) datasets for 
repeatability control. Furthermore, generalisability to patients with rare congenital heart defects was tested 
using the German Competence Network (GCN) dataset.

Our self-supervised approach achieved improved detection accuracy by 30\% - 51\% for SAX and 11\% - 47\% for 4CH  
in ED and ES, as measured by cyclic frame difference (cFD), compared with the volume-based approach.  
We can detect ED and ES, as well as three additional keyframes throughout the cardiac cycle with a mean cFD 
below 1.31 frames for SAX and 1.73 for LAX. Our approach enables temporally aligned inter- and intra-patient 
analysis of cardiac dynamics, irrespective of cycle or phase lengths.

How to get started?
------------
1. **Environment setup**

    Follow the <a target="_blank" href="https://github.com/Cardio-AI/cmr-multi-view-phase-detection/tree/main/docs/Setup.md">Setup Guide</a> to install dependencies and configure your environment.

2. **Data Preparation** 

    See the <a target="_blank" href="https://github.com/Cardio-AI/cmr-multi-view-phase-detection/tree/main/docs/Data.md">Data Guide</a> for instructions on how to structure your dataset.

3. **Run Training and Inference

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
Please cite the following paper (accepted for the @ FIMH2025) if you use/modify or adapt parts of this repository:

**Bibtext**
```
@misc{koehler_mueller_2025-2510.05819,
    author={Sven Koehler and Sarah Kaye Mueller, Jonathan Kiekenap, Gerald Greil, Tarique Hussain, Samir Sarikouch, Florian André, Norbert Frey, Sandy Engelhardt},

  title="Deformable Image Registration for Self-supervised Cardiac Phase Detection in Multi-View Multi-Disease Cardiac Magnetic Resonance Images",
  year="2025",
  eprint={2510.05819},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2510.05819}, 
  abstract="Cardiovascular magnetic resonance (CMR) is the gold standard for assessing cardiac function, but individual cardiac cycles complicate automatic temporal comparison or sub-phase analysis. Accurate cardiac keyframe detection can eliminate this problem. However, automatic methods solely derive end-systole (ES) and end-diastole (ED) frames from left ventricular volume curves, which do not provide a deeper insight into myocardial motion. We propose a self-supervised deep learning method detecting five keyframes in short-axis (SAX) and four-chamber long-axis (4CH) cine CMR. Initially, dense deformable registration fields are derived from the images and used to compute a 1D motion descriptor, which provides valuable insights into global cardiac contraction and relaxation patterns. From these characteristic curves, keyframes are determined using a simple set of rules. The method was independently evaluated for both views using three public, multicentre, multidisease datasets. M&Ms-2 (n=360) dataset was used for training and evaluation, and M&Ms (n=345) and ACDC (n=100) datasets for repeatability control. Furthermore, generalisability to patients with rare congenital heart defects was tested using the German Competence Network (GCN) dataset. Our self-supervised approach achieved improved detection accuracy by 30% - 51% for SAX and 11% - 47% for 4CH in ED and ES, as measured by cyclic frame difference (cFD), compared with the volume-based approach. We can detect ED and ES, as well as three additional keyframes throughout the cardiac cycle with a mean cFD below 1.31 frames for SAX and 1.73 for LAX. Our approach enables temporally aligned inter- and intra-patient analysis of cardiac dynamics, irrespective of cycle or phase lengths."
```


Affiliation
--------
For more Information of our work, please visit our website:
<a target="_blank" href="https://www.klinikum.uni-heidelberg.de/chirurgische-klinik-zentrum/herzchirurgie/forschung/institute-for-artificial-intelligence-in-cardiovascular-medicine-aicm">Institute for Artificial Intelligence in Cardiovascular Medicine </a>

We are part of the Department of Cardiology, Angiology, Pneumology, Heidelberg University Hospital, Heidelberg, Germany
