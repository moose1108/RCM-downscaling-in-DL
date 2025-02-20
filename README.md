## Background
This documentation details the utilization of a CNN-based deep learning script for downscaling climate projections using ERA5 reanalysis data as predictors and TReAD as predictand. The primary goal is to enhance coarse-resolution climate model outputs into high-resolution data, suitable for regional climate studies and assessments of global warming impacts.

## Directory Structure
```
├── Eu/                 # codes adapted from 2024_Bano_Emulators_AIES
├── NZ/                 # codes adapted from high-resolution-downscaling
├── .gitignore
├──README.md
└── environment.yml
```

## Environment
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) if you haven't already.

2. Navigate to the directory containing the `environment.yml` file.

3. Edit the first line of the `environment.yml` file (`name: corrdiff-like`) before creating the environment if you want to create the environment under a different name.

4. Create the environment from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```

## References
- For Eu folder, the code is adapted from [2024_Bano_Emulators_AIES](https://github.com/SantanderMetGroup/2024_Bano_Emulators_AIES/tree/main).
- For NZ folder, the code is adapted from [high-resolution-downscaling](https://github.com/nram812/high-resolution-downscaling). 

## My Work
Find my progress here: [Progress Check](https://docs.google.com/presentation/d/1TNbWKhwYOzwr6m2o73kGC1Jovkl8Xe9D/edit#slide=id.g32cf0b084e1_0_5)