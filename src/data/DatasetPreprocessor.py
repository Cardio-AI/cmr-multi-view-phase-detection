#!/usr/bin/env python3
"""
Script to:
1. Rename and convert medical imaging files to consistent naming convention
2. Generate leave-one-dataset-out cross-validation splits CSV
"""

import os
import re
import pandas as pd
import numpy as np
import shutil

# For medical image conversion
try:
    import nrrd
    import nibabel as nib

    HAS_MEDICAL_LIBS = True
except ImportError:
    HAS_MEDICAL_LIBS = False
    print("Warning: nrrd and/or nibabel not installed. Will install them...")


def install_dependencies():
    """Install required medical imaging libraries"""
    import subprocess
    import sys

    packages = ['pynrrd', 'nibabel']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                               package, '--break-system-packages', '-q'])

    # Re-import after installation
    global nrrd, nib, HAS_MEDICAL_LIBS
    import nrrd
    import nibabel as nib
    HAS_MEDICAL_LIBS = True


def convert_nrrd_to_nifti(nrrd_path, nifti_path):
    """Convert NRRD file to NIfTI format"""
    # Read NRRD file
    data, header = nrrd.read(nrrd_path)

    # Create NIfTI image
    nifti_img = nib.Nifti1Image(data, affine=np.eye(4))

    # Save as NIfTI
    nib.save(nifti_img, nifti_path)
    print(f"Converted: {nrrd_path} -> {nifti_path}")


def process_mnms2_files(input_dir, output_dir):
    """
    Process Dataset 1 (mnms2)
    Pattern: 001_SA_CINE.nii.gz -> mnms2_001_SA.nii.gz
    """
    patients = []
    pattern = re.compile(r'^(\d{3}_SA)_CINE\.nii\.gz$')

    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            patient_name = match.group(1)
            new_filename = f"mnms2_{patient_name}.nii.gz"

            src = os.path.join(input_dir, filename)
            dst = os.path.join(output_dir, new_filename)
            shutil.copy2(src, dst)

            patients.append(f"mnms2_{patient_name}")
            print(f"Copied: {filename} -> {new_filename}")

    return patients


def process_mnms_files(input_dir, output_dir):
    """
    Process Dataset 2 (mnms)
    Pattern: A0S9V9_sa.nii.gz -> mnms_A0S9V9_sa.nii.gz
    """
    patients = []
    pattern = re.compile(r'^([A-Z0-9]{6}_sa)\.nii\.gz$')

    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            patient_name = match.group(1)
            new_filename = f"mnms_{patient_name}.nii.gz"

            src = os.path.join(input_dir, filename)
            dst = os.path.join(output_dir, new_filename)
            shutil.copy2(src, dst)

            patients.append(f"mnms_{patient_name}")
            print(f"Copied: {filename} -> {new_filename}")

    return patients


def process_acdc_files(input_dir, output_dir):
    """
    Process Dataset 3 (acdc)
    Pattern: patient001_4d.nii.gz -> acdc_001_4d.nii.gz
    """
    patients = []
    pattern = re.compile(r'^patient(\d{3}_4d)\.nii\.gz$')

    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            patient_name = match.group(1)
            new_filename = f"acdc_{patient_name}.nii.gz"

            src = os.path.join(input_dir, filename)
            dst = os.path.join(output_dir, new_filename)
            shutil.copy2(src, dst)

            patients.append(f"acdc_{patient_name}")
            print(f"Copied: {filename} -> {new_filename}")

    return patients


def process_gcn_files(input_dir, output_dir):
    """
    Process Dataset 4 (gcn)
    Pattern: 0000-2r1yz1lf_2006-07-31_volume_clean.nrrd -> gcn_0000-2r1yz1lf_2006-07-31.nii.gz
    """
    patients = []
    pattern = re.compile(r'^([0-9a-z\-]+_\d{4}-\d{2}-\d{2})_volume_clean\.nrrd$')

    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            patient_name = match.group(1)
            new_filename = f"gcn_{patient_name}.nii.gz"

            src = os.path.join(input_dir, filename)
            dst = os.path.join(output_dir, new_filename)

            # Convert from NRRD to NIfTI
            convert_nrrd_to_nifti(src, dst)

            patients.append(f"gcn_{patient_name}")

    return patients


def create_leave_one_dataset_out_cv(patients_by_dataset, output_csv):
    """
    Create leave-one-dataset-out cross-validation splits

    Args:
        patients_by_dataset: dict mapping dataset names to list of patient IDs
        output_csv: path to save the CSV file
    """
    datasets = list(patients_by_dataset.keys())
    n_folds = len(datasets)

    rows = []

    # For each fold, one dataset is held out as test
    for fold_idx in range(n_folds):
        test_dataset = datasets[fold_idx]

        for dataset_name, patients in patients_by_dataset.items():
            modality = 'test' if dataset_name == test_dataset else 'train'

            for patient in patients:
                rows.append({
                    'patient': patient,
                    'fold': fold_idx,
                    'modality': modality
                })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Sort for readability
    df = df.sort_values(['fold', 'modality', 'patient']).reset_index(drop=True)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nCreated cross-validation splits CSV: {output_csv}")
    print(f"Total rows: {len(df)}")
    print(f"Number of folds: {n_folds}")
    print(f"\nDataset distribution:")
    for dataset in datasets:
        n_patients = len(patients_by_dataset[dataset])
        print(f"  {dataset}: {n_patients} patients")

    return df


def main():
    """Main execution function"""

    # Check and install dependencies if needed
    if not HAS_MEDICAL_LIBS:
        print("Installing required libraries...")
        install_dependencies()
        print("Libraries installed successfully!\n")

    # Define paths - MODIFY THESE TO MATCH YOUR DIRECTORY STRUCTURE
    base_input_dir = "/mnt/ssd/sarah/data/all_sax/sax/"  # Change this
    output_dir = "/mnt/ssd/sarah/data/all_sax/sax/"  # Change this
    csv_output = "df_kfold.csv"

    # You can also use the current directory
    # base_input_dir = "."
    # output_dir = "./processed_files"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store patients by dataset
    patients_by_dataset = {}

    # Process each dataset
    print("=" * 60)
    print("Processing Dataset 1: mnms2")
    print("=" * 60)
    mnms2_dir = os.path.join(base_input_dir, "mnms2")
    if os.path.exists(mnms2_dir):
        patients_by_dataset['mnms2'] = process_mnms2_files(mnms2_dir, output_dir)
    else:
        print(f"Warning: Directory not found: {mnms2_dir}")
        patients_by_dataset['mnms2'] = []

    print("\n" + "=" * 60)
    print("Processing Dataset 2: mnms")
    print("=" * 60)
    mnms_dir = os.path.join(base_input_dir, "mnms")
    if os.path.exists(mnms_dir):
        patients_by_dataset['mnms'] = process_mnms_files(mnms_dir, output_dir)
    else:
        print(f"Warning: Directory not found: {mnms_dir}")
        patients_by_dataset['mnms'] = []

    print("\n" + "=" * 60)
    print("Processing Dataset 3: acdc")
    print("=" * 60)
    acdc_dir = os.path.join(base_input_dir, "acdc")
    if os.path.exists(acdc_dir):
        patients_by_dataset['acdc'] = process_acdc_files(acdc_dir, output_dir)
    else:
        print(f"Warning: Directory not found: {acdc_dir}")
        patients_by_dataset['acdc'] = []

    print("\n" + "=" * 60)
    print("Processing Dataset 4: gcn")
    print("=" * 60)
    gcn_dir = os.path.join(base_input_dir, "gcn")
    if os.path.exists(gcn_dir):
        patients_by_dataset['gcn'] = process_gcn_files(gcn_dir, output_dir)
    else:
        print(f"Warning: Directory not found: {gcn_dir}")
        patients_by_dataset['gcn'] = []

    # Create cross-validation splits
    print("\n" + "=" * 60)
    print("Creating cross-validation splits")
    print("=" * 60)
    df = create_leave_one_dataset_out_cv(patients_by_dataset, csv_output)

    # Display sample of the dataframe
    print("\nSample of df_kfold.csv:")
    print(df.head(20))

    print("\n" + "=" * 60)
    print("Verification:")
    print("=" * 60)
    for fold in range(len(patients_by_dataset)):
        fold_data = df[df['fold'] == fold]
        test_patients = fold_data[fold_data['modality'] == 'test']['patient'].tolist()
        train_patients = fold_data[fold_data['modality'] == 'train']['patient'].tolist()

        # Get dataset names from patient IDs
        test_datasets = set([p.split('_')[0] for p in test_patients])

        print(f"\nFold {fold}:")
        print(f"  Test dataset(s): {', '.join(test_datasets)}")
        print(f"  Test patients: {len(test_patients)}")
        print(f"  Train patients: {len(train_patients)}")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()