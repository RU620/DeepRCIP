# DeepRCIP
## Overview
DeepRCIP is a deep learning model designed to predict the interaction between RNA and compounds. It takes two inputs: RNA sequences and compound structures (ECFP), and performs binary classification to determine the presence or absence of interaction.
## Requirements
This program has been validated using the following version of the package.
* python 3.8.8
* numpy 1.22.4
* pandas 1.1.5
* matplotlib 3.3.4
* scikit-learn 0.24.0
* rdkit 2023.3.3
* pytorch 1.7.1

## Datasets
All data is in the 'input' folder. 'DataSetA.csv' consists of a comprehensive set of RCIs obtained from our experiments. 'DataSetB.csv' is composed of RCIs registered in [RNAInter](http://www.rnainter.org/), and 'DataSetC.csv' is composed of both. 'MOLInformation.csv' contains information about all compounds that make up interactions in the dataset, while 'RNAInformation.csv' stores information about the RNAs. All datasets are linked by unique identifiers specific to this study, allowing you to access information about the compounds and RNAs that constitute the interactions of interest.

## Program usage
Please execute the following Python files for model training and evaluation. The necessary functions for these executions are stored in the 'src' folder.

 1. 'main.py' performs all the steps for model performance. You can run it by specifying the usage with the 'mode' argument as follows. Additionally, all parameters related to each step are specified in 'config_main.yaml'.

```
$ cd path/to/your/DeepRCIP
$ python main.py --mode init # 
$ python main.py --mode cv #
$ python main.py --mode train #
$ python main.py --mode test #
$ python main.py --mode pi #
```

2. 'visualize.py' draws compound structures highlighted based on the permutation importance of ECFP obtained. All parameters required for execution are specified in 'config_visualize.yaml'.

```
$ cd path/to/your/DeepRCIP
$ python visualize.py
```

3. 'main.ipynb' executes the above code in notebook format. It allows for more flexibility in making changes, and for visualizing compound structures directly.

The results generated from this series of operations will be saved in the 'example' folder.
