# DM-MAGE Project

## Introduction

This project represents Chi's master's thesis work, supervised by Haoj.

**Please note:** Due to their large size, the model checkpoint files are not included in this repository.

## Reproduction Guide

To reproduce the work presented in the associated paper/thesis, please follow these steps:

1.  **Configure CUDA Environment:**
    Ensure that your system has the NVIDIA drivers and the corresponding CUDA toolkit installed correctly. This project relies on GPU acceleration for execution.

2.  **Modify File Paths:**
    The code may contain hardcoded absolute file paths (e.g., for data loading, model saving, etc.). Please review the code and update these paths to reflect the actual locations on your system.

    *(Optional: You can specify here which configuration files or scripts require path modification)*

3. **Run the Code:**
    Execute the main scripts in the following order from the project's root directory. Each step will save the best performing model checkpoint:
    1.  Run the AE (Autoencoder) main script:
        ```bash
        python scDMMGAE-main/ae/main.py
        ```
    2.  Run the GAE (Graph Autoencoder) main script:
        ```bash
        python scDMMGAE-main/gae/main.py
        ```
    3.  Run the pretraining main script:
        ```bash
        python scDMMGAE-main/pretrain/main.py
        ```
    4.  Run the final training main script:
        ```bash
        python scDMMGAE-main/train/main.py
        ```
    *(Adjust script paths/names if they differ from the example)*
## Contact

*2022103618@ruc.edu.cn*
