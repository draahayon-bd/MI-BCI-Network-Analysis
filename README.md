# The Dominant Hand-Area Network: Task-Signal Mismatch and BCI Skill Decoupling
(README.md)
This repository contains the complete Python analysis pipeline for the paper: **"The Dominant Hand-Area Network: Task-Signal Mismatch and the Decoupling of BCI Skill from Global Efficiency in Postural Motor Imagery"**.

The `Master_Script.py` is a menu-driven script that allows for the full reproduction of the paper's results, from raw data preprocessing to final statistical analysis and figure generation.

## 📖 Scientific Overview

This study analyzes the [ds005342 OpenNeuro dataset](https://doi.org/10.18112/openneuro.ds005342.v1.0.3) to investigate the neurophysiology of motor imagery (MI) for Brain-Computer Interfaces (BCIs). The pipeline combines two main analytical frameworks:

1.  **BCI Decoding:** A Common Spatial Pattern (CSP) pipeline with LDA, SVM, and RF classifiers is used to quantify individual BCI skill.
2.  **Network Neuroscience:** The script uses MNE-Python, `mne_connectivity`, and `bctpy` to build functional brain networks (using PLV and Coherence) and directed networks (using Granger Causality). It then analyzes their graph theoretical properties (Global Efficiency, Nodal Strength, Centrality).

The central aim is to test the correlation between local BCI decoding accuracy (from #1) and global network integration (from #2) in both sensor-space and source-space.

## 🛠️ Requirements & Installation

This pipeline is built on the Python scientific stack, primarily using the MNE-Python ecosystem. All required packages are listed in the `environment.yml` file.

To create the exact Conda environment used for this analysis:

1.  **Create the environment:**
    ```bash
    conda env create -f environment.yml
    ```
2.  **Activate the environment:**
    ```bash
    conda activate eeg-mi-analysis
    ```

## 🚀 How to Run

### 1. ⚠️ **Critical Setup Step** ⚠️

Before running, you **must** edit the `Master_Script.py` file.

* Change the `BIDS_ROOT` variable (line 33) to point to the **local path** where you have downloaded the `ds005342` dataset.

    ```python
    # !!! ADJUST THIS PATH !!!
    BIDS_ROOT = r"C:\path\to\your\downloaded\ds005342_dataset"
    ```

### 2. Run the Analysis

Once the `BIDS_ROOT` path is correct and the `eeg-mi-analysis` environment is active, you can run the main script from your terminal:

```bash
python Master_Script.py
