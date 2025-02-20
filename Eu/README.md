## Directory Structure
```
├── train.sh                  # Bash script for launching training
├── train.py                  # Python script for handling the training process
├── inference.py
├── inference.sh
├── utils/
│   ├── deepmodel.py          # Deep learning model structure
│   ├── emulate.py            # Model emulation functions
│   ├── auxiliaryFunctions.py # Other auxiliary functions
├── models/                   # Stores trained models
└── plots/loss/               # Stores loss curve images
```

## Parameter Description
In `train.sh`, the following key variables are defined:
- `variables`: List of meteorological variables (e.g., `w850 q700 q850 t500`).
- `predictand`: The model predicts a specific climate variable. (e.g., `RAINNC`, `T2`).
- `topology`: Model architecture, default is `deepesd`.
- `approach`: Methodology, default is `MOS-E`.
- `start_year`, `end_year`: The training data time range.
- `scale`: Whether to standardize the data.
- `x_data`" Path to self-constructed ERA5 dataset.
- `predictand_data`: Path to predictand dataset.
- `landmask_data`: Path to landmask dataset.
- `loss_path`: Output path for the loss curve.
- `modelPath`: The path where the trained model is saved.

In `inference.sh`, the following key variables are defined:
- `variables`: List of input meteorological variables (e.g., `q700 q850 t500`).
- `predictand`: The target variable to be predicted (e.g., `RAINNC`).
- `topology`: The model architecture (e.g., `deepesd`).
- `approach`: Downscaling methodology (e.g., `MOS-E`).
- `start_year`, `end_year`: Defines the period used for model training.
- `years`: The year for which predictions are generated.
- `scale`: Whether input data is standardized before feeding into the model.
- `bias_correction`: Whether bias correction is applied.
- `modelPath`: Path to the trained model file.
- `predictand_data`: Path to predictand dataset.
- `landmask_data`: Path to landmask dataset.
- `predictor_base`: The base directory for the predictor dataset.
- `predictand_base`: The base directory for the predictand dataset.
- `template_predictand`: The path to a sample predictand file used for structuring the output.
- `outputFileName`: The file path where the inference results are stored.

In `plot.sh`, the following key variables are defined:
- `predict_year`: the year for which the predictions and ground truth will be plotted. In this case, it is set to 2022.
- `model`: the name of the trained model being used for plotting. It is constructed using the topology, variables, training period, inference year, and a version suffix (e.g., "deepesd_q700q850t500t850u200u850v200v850tp_1981_2020_2022_3_b").
- `predictand`: the target variable to be predicted and plotted (e.g., T2 for 2-meter temperature or another variable).
- `predict_path`: the file path to the NetCDF file containing the model predictions. This file is located under the predictions directory for the given predictand and model.
- `gt_path`: the file path to the NetCDF file containing the ground truth data. It is located in the directory with TReAD data for the specified predictand and year.
- `plot_path`: the file path where the generated plot will be saved.
- `landmask_data`: the file path to the landmask NetCDF file that is used to mask out sea areas in the plots.


## Usage
### 1. Start Training
```bash
bash train.sh
```
- This command runs `train.py` using the variables defined in `train.sh`. 
- After training, the following files are generated:
    - The trained model is stored in `./models/`.
    - The loss curve image is saved in `./plots/loss/`.
### 2, Running Inference
```bash
bash inference.sh
```
- This command runs `inference.py` using the variables defined in `inference.sh`. 
- After inferencing, the following file is generated:
    - NC file for the model's inference.

## Results
- Evaluation metrics for prediction on 2022:
    |  | Corr | RMSE | MAE |
    |----------|----------|----------|----------|
    | RAINNC   |  0.48  |  8.84  | 4.49 |
    | T2   |  0.993  |  0.828  | 0.690 |


## Notes
- Ensure `train.py` can access the modules inside `utils/`.
- `predictor_data` and `predictand_data` paths must be correct.
- `landmask_data` must match `predictand_data`.