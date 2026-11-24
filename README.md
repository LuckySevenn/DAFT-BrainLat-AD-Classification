# Multimodal Alzheimer's Disease Classification with DAFT (BrainLat)

This repository contains the custom scripts and notebooks of my degree project (Final Career Project, Systems Engineering, Universidad del Valle). The project compares multimodal and single-modality models for classifying **cognitively normal (CN)** vs **Alzheimer's disease (AD)** participants using T1-weighted MRI and tabular data from the **BrainLat** dataset, with an external test on a subset of **ADNI**.

The DAFT model is **not** reimplemented or modified here. It was used through its original public implementation ([ai-med/DAFT](https://github.com/ai-med/DAFT)). This repository contains the additional scripts written to prepare the data, call DAFT to run the experiments, and evaluate the models.

Models compared (all on the same stratified BrainLat partitions):

- **DAFT** (Dynamic Affine Feature Map Transform): MRI + tabular data
- **3D ResNet** (image only, DAFT's backbone without the DAFT module)
- **FT-Transformer** (tabular only)
- **Random Forest** and **XGBoost** (tabular baselines)

The pipeline performs:

1. Visual quality control of the T1-weighted scans
2. FreeSurfer processing (recon-all + hippocampal subfields and amygdala nuclei)
3. Extraction of 16 morphometric features, eTIV adjustment and construction of the master CSV with the data split
4. Extraction of a 64 × 64 × 64 patch centred on the left hippocampus and conversion to HDF5 (the input format of DAFT)
5. Hyperparameter grid search for DAFT and the 3D ResNet
6. Training of FT-Transformer, Random Forest and XGBoost (Google Colab notebooks)
7. Evaluation of the final models on the ADNI test subset

## Repository structure

```
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── 01_quality_control/
│   │   └── mri_snapshots.py           # Sagittal, coronal and axial views of each scan
│   ├── 02_freesurfer/
│   │   └── PreProcessing.py           # FreeSurfer batch (BrainLat and ADNI)
│   ├── 03_tabular_features/
│   │   ├── build_master_csv.py        # BrainLat master CSV with the train/valid/test split
│   │   └── build_master_csv_adni.py   # ADNI master CSV (all subjects in test)
│   ├── 04_hdf5/
│   │   ├── build_hdf5.py              # BrainLat train.h5, valid.h5, test.h5
│   │   └── build_hdf5_adni.py         # ADNI test.h5 (normalized with BrainLat train statistics)
│   ├── 05_training/
│   │   ├── run_grid.py                # Grid search for DAFT and the 3D ResNet
│   │   └── daft_plots.py              # Learning curves, confusion matrices, ROC and metrics
│   └── 06_evaluation_adni/
│       └── evaluate_adni.py           # Evaluation of the final DAFT and ResNet models on ADNI
├── notebooks/
│   ├── ftt_optimizadores_colab.ipynb  # FT-Transformer
│   ├── rf_xgboost_colab.ipynb         # Random Forest and XGBoost
│   └── descriptivos_brainlat_adni.ipynb  # BrainLat vs ADNI cohort comparison (boxplots, demographic table)
└── configs/
    └── final_configurations.yaml      # Final hyperparameters of each model
```

## Data

The MRI images and clinical data are **not included** in this repository. BrainLat (Synapse) and ADNI must be requested from their owners under their data use agreements.

- BrainLat: 343 participants (137 CN, 206 AD) with complete clinical data and complete FreeSurfer outputs.
- ADNI: 100 participants from ADNI-2 (50 CN, 50 AD), baseline T1, 3T. Used only for testing.
- Tabular input (19 variables): age, sex, years of education and 16 bilateral morphometric measures (hippocampal volume, lateral ventricle volume, entorhinal, inferior temporal and parahippocampal cortical thickness, CA1 and subiculum volume, basolateral amygdala volume). Volumes are divided by eTIV.

| Cohort   | Partition   | CN | AD | Total |
|----------|-------------|----|----|-------|
| BrainLat | Training    | 96 | 144 | 240 |
| BrainLat | Validation  | 21 | 31 | 52 |
| BrainLat | Test        | 20 | 31 | 51 |
| ADNI     | Test subset | 50 | 50 | 100 |

## Instructions

Each script has a CONFIGURATION section at the top with the paths to set before running it.

1. Clone this repository and the original DAFT repository, and install the dependencies (`pip install -r requirements.txt`). FreeSurfer 7.4.1 with the hippocampal subfields module is needed for step 3.

        git clone <this-repository>
        git clone https://github.com/ai-med/DAFT.git

2. **Quality control.** Run `scripts/01_quality_control/mri_snapshots.py` on the folder with the T1-weighted scans and review the images.
3. **FreeSurfer.** Run `scripts/02_freesurfer/PreProcessing.py`: `INPUT_MODE = "csv"` for BrainLat (once per country) and `INPUT_MODE = "folder"` for ADNI.
4. **Master CSV.** Run `scripts/03_tabular_features/build_master_csv.py` (BrainLat) and `build_master_csv_adni.py` (ADNI).
5. **HDF5.** Run `scripts/04_hdf5/build_hdf5.py` (BrainLat) and then `build_hdf5_adni.py` (ADNI, which needs the BrainLat `train.h5`).
6. **DAFT and 3D ResNet.** Copy `run_grid.py` and `daft_plots.py` from `scripts/05_training/` to the root of the DAFT repository (next to `train.py`) and run `python run_grid.py` from there.
7. **FT-Transformer, Random Forest and XGBoost.** Open the notebooks in `notebooks/` in Google Colab and upload the BrainLat (and ADNI) master CSV when asked.
8. **ADNI evaluation.** Copy `scripts/06_evaluation_adni/evaluate_adni.py` to the root of the DAFT repository (next to `run_grid.py` and `daft_plots.py`) and run `python evaluate_adni.py`.
9. **Cohort comparison.** Open `notebooks/descriptivos_brainlat_adni.ipynb` in Google Colab to reproduce the boxplots and the demographic table.

## Notes

- The original DAFT code is used without changes; the scripts in this repository only call it.
- For DAFT, the ResNet and the FT-Transformer, each run keeps the checkpoint with the highest validation balanced accuracy.
- The final configuration of each model (`configs/final_configurations.yaml`) is the one with the highest validation balanced accuracy among those whose validation–test balanced-accuracy gap is less than 10 percentage points. Because the BrainLat test set was used in this selection, its results should be read as optimistic.
- The ADNI tabular variables and images are normalized with the statistics of the BrainLat training set.
- DAFT encodes the AD class as `"Dementia"`, so the master CSVs and HDF5 files use that label.

## References

**Datasets**

BrainLat: P. Prado, V. Medel, R. Gonzalez-Gomez, A. Sainz-Ballesteros, V. Vidal, H. Santamaria-Garcia, S. Moguilner, J. Mejia, A. Slachevsky, M. I. Behrens, D. Aguillon, F. Lopera, M. A. Parra, D. Matallana, M. A. Maito, A. M. Garcia, N. Custodio, A. Ávila Funes, S. Piña-Escudero, A. Birba, S. Fittipaldi, A. Legaz, and A. Ibañez, "The BrainLat project, a multimodal neuroimaging dataset of neurodegeneration from underrepresented backgrounds," Scientific Data, vol. 10, p. 859, 2023. DOI: [10.1038/s41597-023-02774-3](https://doi.org/10.1038/s41597-023-02774-3)

BrainLat data: BrainLat Project, "The BrainLat project, a multimodal neuroimaging dataset of neurodegeneration from underrepresented backgrounds," Synapse repository, 2023. DOI: [10.7303/syn51549340](https://doi.org/10.7303/syn51549340). Licensed under CC BY 4.0.

ADNI: Alzheimer's Disease Neuroimaging Initiative. https://adni.loni.usc.edu/

**Methods and tools**

DAFT: T. N. Wolf, S. Pölsterl, and C. Wachinger, "DAFT: A universal module to interweave tabular data and 3D images in CNNs," NeuroImage, vol. 263, p. 119505, 2022. DOI: [10.1016/j.neuroimage.2022.119505](https://doi.org/10.1016/j.neuroimage.2022.119505). Code: https://github.com/ai-med/DAFT

FT-Transformer: Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, "Revisiting deep learning models for tabular data," in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 18932–18943.

FreeSurfer: B. Fischl, "FreeSurfer," NeuroImage, vol. 62, no. 2, pp. 774–781, 2012. DOI: [10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)

Random Forest: L. Breiman, "Random forests," Machine Learning, vol. 45, no. 1, pp. 5–32, 2001. DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)

XGBoost: T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016, pp. 785–794. DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
