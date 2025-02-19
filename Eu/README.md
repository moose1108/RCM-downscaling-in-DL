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
- `predictand`: The target variable (e.g., `RAINNC`, `T2`).
- `topology`: Model architecture, default is `deepesd`.
- `approach`: Methodology, default is `MOS-E`.
- `start_year`, `end_year`: The training data time range.
- `scale`: Whether to standardize the data.
- `predictor_data`, `predictand_data`: Paths to predictor and predictand datasets.
- `landmask_data`: Path to landmask dataset.
- `loss_path`: Output path for the loss curve.

In `inference.sh`, the following key variables are defined:

- `variables`: List of input meteorological variables (e.g., `q700 q850 t500`)
- `predictand`: The target variable to be predicted (e.g., `RAINNC`)
- `topology`: The model architecture (e.g., `deepesd`)
- `approach`: Downscaling methodology (e.g., `MOS-E`)
- `start_year`, `end_year`: Defines the period used for model training
- `years`: The year for which predictions are generated
- `scale`: Whether input data is standardized before feeding into the model
- `bias_correction`: Whether bias correction is applied
- `modelPath`: Path to the trained model file

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
    | T2   |  0.994  |  0.701  | 0.573 |


## Notes
- Ensure `train.py` can access the modules inside `utils/`.
- `predictor_data` and `predictand_data` paths must be correct.
- `landmask_data` must match `predictand_data`.