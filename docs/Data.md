Dataset
------------
For this project we used cine-SSFP CMR images from several publicly available sources:
- Multi-Disease, Multi-View & Multi-Center Right Ventricular Segmentation in Cardiac MRI (<a target="_blank" href="https://www.ub.edu/mnms-2/">M&Ms-2</a>)
  - 2D+t 4CH and 3D+t SAX
  - Training and Evaluation
  - public
- Automated Cardiac Diagnosis Challenge (<a target="_blank" href="https://www.creatis.insa-lyon.fr/Challenge/acdc/">ACDC</a>)
  - 3D+t SAX
  - Evaluation
  - public
- Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation Challenge (<a target="_blank" href="https://www.ub.edu/mnms/">M&Ms</a>)
  - 3D+t SAX
  - Evaluation
  - public
- German Competence Network (<a target="_blank" href="https://www.clinicaltrials.gov/study/NCT00266188?term=NCT00266188&rank=1">GCN</a>)
  - 2D+t 4CH and 3D+t SAX
  - Evaluation
  - private

The training of both models, **registration** and **segmentation**, was performed on the 200 datasets from the M&Ms-2 **Training set**.
Evaluation was performed on the **Testing set** and all other datasets.
For M&Ms-2 Testing set, ACDC and GCN were annotations of 5 keyframes provided by experienced physicians.
These manual labels include:
- End-diastole (ED)
- mid-systole (MS): maximum contraction resulting in a peak ejection between ED and ES
- end-systole (ES)
- peak flow (PF): peak early diastolic relaxation
- mid-diastole (MD): phase before atrial contraction at the on-set of the p-wave

Please contact us if you are interested in these labels.

### Structure

The project expects a single data-root folder.
This folder must contain at least oner of the following subfolders
- ```lax ``` for 4CH cine CMR
- ```sax```  for SAX cine CMR 

The folder name must be specified int the ``"view"`` field of ``dataset.json``.
Please use **"sax"** if you are working with 3D+t image stacks (e.g., SAX CMR).
For 2D+t data, you can choose any naming convention, but we recommend **"lax"**.

Optionally, it may contain a csv file for splitting the data **df_kfold.csv** and one with the groundtruth keyframe annotations.

 ```
   ├── lax/             # folder with cine 4CH CMR
   ├── sax/             # folder with cine SAX CMR
   ├── dataset.json     # Json with all necessary information about labels, suffix and post processing
   ├── df_kfold.csv     # dataframe with split for k-fold split validation
   └── phases.csv       # dataframe with keyframes for calculation of cyclic frame difference
 ```

### Cine CMR
If you want to use your own data:
- Accepted formats:  NRRD ( ```nrrd ```) or NIfTI ( ```.nii.gz ```)
- Expected shape:
    ```
    t × x × y × z
    ```
  where ```z = 1``` for 4CH images (single slice), and ```z = n``` for SAX.
- Temporal dimension: set in the config as ```"T_SHAPE" ```.
  - Default = 40 (i.e, up to 39 frames).
  - If your sequence has more than 39 frames, increase ```"T_SHAPE" ``` accordingly (t = max. sequence length + 1).

#### File naming conventions
File names are used to extract patient IDS, which must match those in phases.csv and df_kfold.csv.

Supported regex patterns (from source datasets):
- ```r'\d+-([a-zA-Z0-9]+)_\d{4}-\d{2}-\d{2}.*'```  -> GCN: 0000-0ae4r74l_1900-01-01_...
- ```r'(\d+)_LA_CINE.*'```  -> M&Ms and M&Ms-2: 039_LA_CINE.nii.gz
- ```r'patient(\d+)_.*'```  -> ACDC: patient001_4d.nii.gz

If your dataset uses a different convention, add your regex pattern to:
```src/data/Dataset.py -> extract_id```.

### Metadata files

#### ``phases.csv`` - Ground-truth keyframes
- Required for evaluation (cyclic frame difference calculation).
- Format: one row per patient, one column per keyframe
- Example
    ``` 
    patient, ED#, MS#, ES#, PF#, MD#
    001_LA_CINE.nii.gz, 24, 3, 9, 12, 21
    002_LA_CINE.nii.gz, 0, 4, 9, 13, 22
    ...
    ``` 

#### ``df_kfold.csv`` - K-fold split definition
- Required if you want cross-validation
- Format: one row per patient per fold. Must include patient ID, fold index, and split assignment (train/test)
- Example (4-fold):
    
  ``` 
  patient, fold, modality
  001_LA, 0, train
  002_LA, 0, train
  003_LA, 0, train
  004_LA, 0, test
  ...
  001_LA, 1, train
  002_LA, 1, test
  003_LA, 1, train
  004_LA, 1, train
  ...
  001_LA, 2, train
  002_LA, 2, train
  003_LA, 2, test
  004_LA, 2, train
  ...
  001_LA, 3, train
  002_LA, 3, train
  003_LA, 3, train
  004_LA, 3, test
  ...
  ``` 
#### ``dataset.json`` - Dataset description
This file standardizes dataset handling across scripts.
It defines:
- Labels and their numeric encoding
- File suffix and endings 
- Post-processing steps.

##### Example:
  ```
  { 
     "channel_names": {
       "0": "cineMRI"
     }, 
     "labels": { 
        "background": 0,
        "LV": 1,
        "MYO": 2,
        "RV": 3
     }, 
     "suffix": {
        "image_suffix": "CINE",
        "mask_suffix": "", 
        "file_ending": ".nii.gz"	
     },
    "view":"LAX",
    "use_segmentation": false,
     "start_id": 0,
     "post_processing":{
         "focus_point": "MSE",
         "norm_threshold": 40,
         "cc_filter": None,
         "use_segmentation": false,
         "mask_channels": null
         }
     }
  ```

**Key points**
  
- ``"view"`` must match the folder name containing your data.
  - use ``"sax"`` for 3D+t image stacks
  - For 2D+t data you can use any folder name
- When specifying file suffixes in ``dataset.json`` (e.g., "image_suffix": "CINE"), 
ensure filenames follow the same pattern so that image–mask matching works reliably (see Cine CMR - File naming conventions).
- **Post-processing parameters**
  - ``"focus_point"``:
    - ``"MSE"``: center of mass of self-supervised mask
    - ``"VOL"``: center of the whole volume/image
    - ``"Septum"`` or label indices (e.g. ``[1]`` for LV cavity, ``[1,2,3]`` for whole heart) if segmentation is available
  - ``"norm_threshold"``: Percentile of norm threshold (0-100)
  - ``"cc_filter"``: connected component filtering (``null`` to disable)
- Segmentation settings:
  - ``"use_segmentation": true``: masks the data using segmentation labels instead of self-supervised masking
  - If ``"mask_channels"`` is ``null`` or ``[]``, all labels (except "background") are used.
  - To restrict masking, specify label IDS explicitly (e.g. ``[2]`` for LV myocardium)
  

