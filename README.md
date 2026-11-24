# Multimodal Alzheimer's Disease Classification with DAFT (BrainLat)

This repository contains the code of my degree project (Final Career Project, Systems Engineering, Universidad del Valle). It compares multimodal and single-modality models for classifying **cognitively normal (CN)** vs **Alzheimer's disease (AD)** subjects using T1-weighted MRI and tabular data from the **BrainLat** dataset, with an external test on a subset of **ADNI**.

Models compared (all on the same stratified 70/15/15 split):

- **DAFT** (Dynamic Affine Feature Map Transform): MRI + tabular data
- **ResNet** (image only)
- **FT-Transformer** (tabular only)
- **Random Forest** and **XGBoost** (tabular baselines)

The pipeline performs:

- Cortical and subcortical segmentation with FreeSurfer
- Construction of a master CSV with tabular features and the data split
- Conversion of MRI + tabular data to HDF5 (the format DAFT reads)
- Hyperparameter grid search for DAFT and ResNet
- Training of FT-Transformer, Random Forest and XGBoost (Google Colab notebooks)
- External evaluation of the trained models on ADNI

## Repository structure

```
.
├── README.md
├── requirements.txt
├── configs/                  # Final hyperparameter configuration of each model
├── daft/                     # Copy of the DAFT repository (with the modified cli.py)
├── data/
│   ├── brainlat/             # BrainLat master CSV and images (not included)
│   ├── adni/                 # ADNI master CSV and images (not included)
│   └── hdf5/                 # Generated HDF5 files (not included)
├── notebooks/
│   ├── rf_xgboost_colab.ipynb
│   └── ftt_optimizadores_colab.ipynb
├── scripts/
│   ├── preprocessing/        # FreeSurfer batch, CSV building, HDF5 conversion
│   ├── training/             # Grid search over DAFT's train.py
│   └── evaluation/           # Evaluation of trained models on ADNI
├── results/                  # Summary of training results
└── docs/                     # Extra documentation
```

## Data

The MRI images and clinical data are **not included** in this repository. BrainLat and ADNI must be requested from their owners under their data use agreements.

- Only BrainLat subjects with `valid == 1` in `BrainLat_Valid.csv` are used.
- Final sample: 137 CN / 206 AD.
- Tabular input (19 variables): age, sex, years of education and 16 bilateral FreeSurfer measures (hippocampus, ventricles, cortical thickness and hippocampal subfields).
- ADNI external test: 100 subjects (50 CN + 50 AD, baseline, 3T), processed with the same FreeSurfer script. All ADNI subjects are used as test.

## Instructions

1. Clone the repository.
   ```
   git clone <this-repository>
   cd <this-repository>
   ```
2. Install the dependencies.
   ```
   pip install -r requirements.txt
   ```
   Install PyTorch following the [official instructions](https://pytorch.org/get-started/locally/).
3. Run FreeSurfer on the MRI scans with the script in `scripts/preprocessing/`.
4. Build the master CSV (features + `split` column) and place it in `data/brainlat/` (and `data/adni/` for ADNI).
5. Convert the data to HDF5 with the conversion script. The files are written to `data/hdf5/`.
6. Run the grid search for DAFT and ResNet from `scripts/training/`.
7. Open the notebooks in `notebooks/` in Google Colab to train FT-Transformer, Random Forest and XGBoost using the same master CSV.
8. Evaluate the trained models on ADNI with the scripts in `scripts/evaluation/`.

## Notes

- The only change to the original DAFT code is in `daft/cli.py`: the SGD optimizer also receives weight decay.
- Final configurations were selected as the highest validation balanced accuracy among runs whose validation–test balanced-accuracy difference was strictly less than 10 percentage points. Ties were broken by that difference, then by F1 and AUC. Because of this, the reported test results should be read as optimistic.
- The full training results are in `Datos_entrenamiento.xlsx` (see `results/`).

## References

DAFT: T. N. Wolf, S. Pölsterl, and C. Wachinger, "DAFT: A universal module to interweave tabular data and 3D images in CNNs," NeuroImage, 260:119505, 2022. Code: https://github.com/ai-med/DAFT

FT-Transformer: Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, "Revisiting deep learning models for tabular data," NeurIPS, 2021.

FreeSurfer: B. Fischl, "FreeSurfer," NeuroImage, 62(2):774–781, 2012.

XGBoost: T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," KDD, 2016.

Random Forest: L. Breiman, "Random forests," Machine Learning, 45(1):5–32, 2001.
