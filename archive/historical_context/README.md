# EEG Foundation Model Paper - Dataset

## Dataset Overview

This project contains Visual Evoked Potential (VEP) EEG data with preprocessing pipeline.

### Total Files: 63 recordings

## Dataset Breakdown by Category

| Category | Phase 1 | Phase 2 | Total |
|----------|---------|---------|-------|
| Apple | 7 (A1) | 7 (A2) | 14 |
| Car | 8 (C1) | 8 (C2) | 16 |
| Flower | 8 (F1) | 9 (F2) | 17 |
| Human Face | 8 (P1) | 8 (P2) | 16 |
| **TOTAL** | | | **63** |

### Flower F2 Subjects (9 files)
- sub4, sub5, sub7, sub9, sub13, sub17, sub18, sub19, sub28

## Directory Structure

```
VEP-EDF/                          # Raw EDF files (63 files)
EEGLAB-SET/                       # Converted .set files (63 files)
EEGLAB-SET_STEP1_ICA/            # ICA cleaned files (63 files)
EEGLAB-SET_STEP2_FILTERED/       # Re-referenced + filtered files (63 files)
EEGLAB-SET_STEP3_EPOCHED/        # Epoched files (63 files, 3,612 total epochs)
CSV-FEATURES/                     # CSV files for ML training (63 files)
```

## Preprocessing Pipeline

### Step 1: ICA + ICLabel Artifact Removal
- **Script**: `Preprocess.m`
- **Input**: `EEGLAB-SET/`
- **Output**: `EEGLAB-SET_STEP1_ICA/`
- **Process**:
  - Runs Independent Component Analysis (ICA)
  - Uses ICLabel to classify components
  - Removes Eye and Muscle artifacts (≥90% probability)
  - Resamples to 128 Hz if sampling rate < 100 Hz
- **Report**: `iclabel_removal_report.csv`
- **Results**: 63/63 files processed successfully

### Step 2: Re-referencing + Bandpass Filtering
- **Script**: `Preprocess_Step2.m`
- **Input**: `EEGLAB-SET_STEP1_ICA/`
- **Output**: `EEGLAB-SET_STEP2_FILTERED/`
- **Process**:
  - Average re-referencing (common average reference)
  - Bandpass filter: 0.1 - 50 Hz
  - High-pass (0.1 Hz): Removes DC drift and baseline wander
  - Low-pass (50 Hz): Removes high-frequency noise
- **Report**: `filtering_report.csv`

### Step 3: Epoching
- **Script**: `Preprocess_Step3.m`
- **Input**: `EEGLAB-SET_STEP2_FILTERED/`
- **Output**: `EEGLAB-SET_STEP3_EPOCHED/`
- **Process**:
  - Sliding window epoching (1 second windows)
  - 50% overlap between consecutive windows
  - Artifact rejection: ±150 µV threshold
  - Total epochs extracted: 3,612 epochs
- **Report**: `epoching_report.csv`

### Step 4: Feature Extraction & CSV Export
- **Script**: `Preprocess_Step4_ExportCSV.m`
- **Input**: `EEGLAB-SET_STEP3_EPOCHED/`
- **Output**: `CSV-FEATURES/`
- **Process**:
  - Exports each epoch to CSV format
  - Feature dimensions: 14 channels × 128 timepoints = 1,792 features
  - Includes metadata: Class, Phase, Subject
- **Output Files**: 63 CSV files with epoch data

## Machine Learning Results

### SVM Classification (4-Class)
- **Classes**: Apple, Car, Flower, Human Face
- **Method**: Stratified 5-Fold Cross-Validation
- **Kernel**: RBF (Radial Basis Function)
- **Features**: Raw time-domain EEG data (1,792 features)
- **Total Samples**: 3,612 epochs

#### Performance Metrics

| Metric | Mean | Std Dev |
|--------|------|---------|
| **Accuracy** | **41.86%** | ±0.87% |
| **F1-Score (Macro)** | **37.41%** | ±1.53% |
| **F1-Score (Weighted)** | **39.40%** | ±1.38% |
| **Precision (Macro)** | **43.56%** | ±1.01% |
| **Recall (Macro)** | **38.51%** | ±1.11% |

#### Per-Class Accuracy
- 🌸 **Flower**: 58.78% (best performance)
- 👤 **Human Face**: 55.38%
- 🚗 **Car**: 25.34%
- 🍎 **Apple**: 14.55% (most challenging)

#### Class Distribution
- Apple: 646 epochs (17.9%)
- Car: 872 epochs (24.1%)
- Flower: 1,099 epochs (30.4%)
- Human Face: 995 epochs (27.5%)

**Note**: Results represent baseline performance using raw EEG features. Performance can be improved with:
- Advanced feature engineering (PSD, wavelet transforms, CSP)
- Deep learning architectures (CNN, Transformer)
- Hyperparameter optimization
- Data augmentation techniques

## Technical Details

- **Channels**: 14 EEG channels (Emotiv EPOC-X)
- **Channel Layout**: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
- **Sampling Rate**: 128 Hz (after preprocessing)
- **Data Format**: EEGLAB .set format

## Usage

### Running Preprocessing

```matlab
% Step 1: ICA + ICLabel Artifact Removal
Preprocess

% Step 2: Re-referencing + Bandpass Filtering
Preprocess_Step2

% Step 3: Epoching
Preprocess_Step3

% Step 4: Export to CSV
Preprocess_Step4_ExportCSV
```

### Running SVM Classification

```bash
# Open Jupyter Notebook
jupyter notebook SVM_Classification.ipynb

# Or run all cells programmatically
```

### Importing New EDF Files

```matlab
ImportRawToSET
```

## Notes

- All 63 original .edf files successfully converted to .set format
- Step 1 successfully processed all 63 files
- 2 files (Apple_A2_sub23, Apple_A2_sub25) required resampling due to low sampling rate
- ICLabel removed eye artifacts from approximately 14 files
