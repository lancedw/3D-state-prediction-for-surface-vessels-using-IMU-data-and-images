# 3D state prediction for surface vessels using IMU data and images - a deep learning approach
2022-2023 Thesis by Lance De Waele   

**Supervisors**:  Prof. dr. ir. Hiep Luong, Prof. dr. ir. Jan Aelterman  
**Counsellors**:  Ir. Tien-Thanh Nguyen (Royal Military Academy), Dr. ir. Benoit Pairet (Royal Military Academy)  

In co-op. with the Royal Military Academy of Belgium  

# Layout of repository
```
.
├── windows_eny.yml         # Anaconda environment for execution on windows
├── docs/      		    # Presentations, paper, worklog, workplan, images
│   └── ...          
├── code/
│   ├── 3dmodel/                # contains simulation data*
│   ├── results/                # contains saved graphical .png results from training and testing
│   ├── Notebooks/              # source code
|   │   ├── data_loaders/              # contains .py files with classes for data loading, sequencing and splitting
|   │   ├── model_states/              # contains binary files with state dictionaries for each trained model
|   │   ├── models/                    # contains .py files for each model and a model_provider.py to easily access them
|   │   ├── test_notebooks/            # contains notebooks for model testing 
|   │   ├── test_results/              # contains binary files with the MSEs of each model on all test sequences
|   │   ├── train_notebooks/           # contains one notebook for each model with complete training and testing functionalities 
|   │   ├── training_results/          # contains binary files with the training and validation MSE losses
|   │   ├── pr_data_analysis.ipynb            # notebook containing all code for data analysis
└── └── └── train_results_plots.ipynb         # notebook where all plots are made for training and validation loss
``` 
*simulation data can be downloaded here: https://drive.google.com/drive/folders/1RF8_wFfcIM0GIklXflPYv-tK3uaEWSSZ

# Execution of the code
To execute the notebooks, you can create a conda environment with the .yml file (Windows)

Alternatively: run the first cell of any notebook and manually import all the packages that are used in the notebook. Python 3.8 was used.

If GPU support is desired, Pytorch should be installed in the conda environment with the cuda toolkit extensions. To do this, follow the link to the PyTorch website, select your OS and cuda version, and run the provided command in the environment's terminal. 
https://pytorch.org/get-started/locally/
(If your GPU only supports cuda 10.2, you will need to install a older version of pytorch)
