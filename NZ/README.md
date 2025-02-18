## Directory Structure
```
├── train.py                  # Python script for handling the training process
├── inference.py
├── src/
│   ├── losses.py             # Deep learning model structure
│   ├── models.py             # Model emulation functions
│   ├── prepare_data.py       # Other auxiliary functions
└── models/                   # Stores trained models
```


**Running Scripts**:
Each script can be executed independently, depending on the stage of the project:
- `python train.py` to train the model.
- `python inference.py` to generate predictions.
- `python MAE.py` to calculate error metrics such as MAE and RMSE and generate figures.
- `python plot.py` for visualization given selected days.