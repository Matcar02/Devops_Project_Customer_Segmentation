# Customer Segmentation MLOps Project
#### An end-to-end work to manage a customer segmentation ML project using MLOps principles. 

## Description
### The project allows business stakeholders to visualize and inspect customer segmentations using clustering methods. This method allows the business to save money by identifying the best products to sell as well as the consumers' profile and their spending behaviour to enhance marketing strategies such as cross-selling.
### The methods provide relevant metrics such as Silhouette score, which indicates the proximity (how close customers are) of the elements in the segments to assess similarities in consumers' behaviour. 
### The use of MLops packages such as MLFlow and W&B provides simple and effective ways to observe, monitor, test, automate and deploy improved and reliable Clustering Models that are ready for Data Drifts and Anomalies, as well as new Consumer Behaviour' patterns.

## Installation
1. Clone the repository.
2. Create a conda environment using the conda.yml file with this command
```
conda env create -f conda.yml
```
3. Activate the conda env:
```
conda activate devops_env
```


## Project Structure (Tree)
The project uses a config.yaml for configuration settings, including data paths and model parameters.
------------
```

├───data          <- data used for modeling.
│   ├───external  <- csv files to use.
│   ├───interim
│   ├───processed
│   └───raw
├───docs        
├───models
├───notebooks     <- contains notebooks including all code and notebook to run every module.
├───references  
├───reports       <- this will contain reports made throughout the projects (graphics, plots, stats etc...). 
│   └───figures
└───src
    ├───clustering   <- contains all the clustering modules and functions (models).
    │   ├───agglomerative
    │   │   └───__pycache__
    │   ├───ann_methods
    │   │   └───__pycache__
    │   ├───kmeans
    │   │   └───__pycache__
    │   ├───pca_methods
    │   │   └───__pycache__
    │   └───spectral
    │       └───__pycache__
    ├───data_preparation     <- contains all modules for data preparation and cleaning.
    │   └───__pycache__
    ├───descriptive_stats    <- contains descriptive stats on customer segments
    │   └───__pycache__
    ├───dimensionality_reduction  <- contains modules to apply dimensionality reduction for PCA
    │   └───__pycache__
    ├───utilities       
    ├───visualization    <- contains modules that produce plots and relevant graphics
    │   └───__pycache__
    └───__pycache__

```
## Use
In order to use and test the code, please run the different modules and functions in the notebook folder either under ```checkall.ipynb``` or another python file you may want to use. The paths have been set such that the files are stored and processed in specific paths. All the artifacts and output files are stored in the ```reports``` folder where are divided into figures (plots) and dataframes (csv files).

## Testing 
In order to carry out tests on the code, make sure to download pytest. Once you have it installed, run:

```
pytest tests/
```

## License 
The project is licensed under the MIT License.
