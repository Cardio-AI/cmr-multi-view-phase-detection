Training
------------
For self-supervised deformable image registration and keyframe detection, you only need to train:
- **Deformable image registration model** (self-supervised, no groundtruth annotations necessary)

## Deformable image registration model
Our trainings script supports single- and multi-GPU training (local or  cluster). 
You can either run the training  using the jupyter notebook in ```notebooks/Train/Train_cv.ipynb``` (recommended) or run the script from your command line or preferred IDE.

For training you need:
   - Root folder with 4d (sax) 3d (4CH) nrrd or nii.gz files (4CH single slice cine CMR; see <a target="_blank" href="https://github.com/Cardio-AI/cmr-multi-view-phase-detection/tree/main/docs/Data.md">data/Data</a>)
   - Metadatafiles:
      - phases.csv: Dataframe with keyframes for calculation of cyclic frame difference
      - df_kfold.csv (optional): Dataframe with split for k-fold split validation
 
### For training with the notebook
A step-by-step guide is included in the notebook itself.
In the notebook you will create the additional files necessary for training: 
   - ```dataset.json```: Dataset json with all necessary information about labels, suffix and post processing
   - ```config.json```: Config file with input dimensions, GPU, k-fold training, 

### For training from your console/preferred IDE:
1. **Config setup**
   - Start from or modify an example config in <a target="_blank" href="https://github.com/Cardio-AI/cmr-multi-view-phase-detection/tree/main/data/configs">data/configs</a>
2. **Run Training**
      ```
    python src/models/train_regression_model.py \
   - cfg_reg <path_to_config> \
   - data_json <path_to_dataset_json> \
   - data <data_root> \
   - inmemory <true/false>         
    ```
    -  ```cfg_reg ```: Path to an experiment config (examples in ```data/configs```)
    - ```data```: Root folder with 4d (sax) 3d (4CH) nrrd or nii.gz files
    - ```data_json```: Path to dataset.json, which contains all necessary information about labels, suffix and post-processing  (examples in ```data/configs```).
    - ```inmemory```: Enables in-memory pre-processing for cluster-based trainings

3.  **Cross Validation**
    - Script can train on multiple folds sequentially (defined in config ```"FOLDS":[0, 1, 2, 3],```). For multi-fold splits, you must supply a **_df_kfold.csv_** file (see <a target="_blank" href="https://github.com/Cardio-AI/cmr-multi-view-phase-detection/tree/main/docs/Data.md">data/Data</a>)
    - If no folds are provided, the model is trained on all available data
    - After training, predictions are automatically saved into f0-f3 subfolders. 

4.  **Outputs per fold**
   ```
   ├── config (config used in this experiment fold)
   ├── Log_errors.log (logging.error logfile)
   ├── Log.log (console and trainings progress logs, if activated
   ├── model (model graph and weights for later usage
   ├── model.png (graph as model png)
   ├── model_summary.txt (layer input/output shapes as structured txt file
   └── tensorboard_logs (tensorboard logfiles: train-/test scalars and model predictions per epoch)
   ```
