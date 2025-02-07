# Vehicle State Model
## Installation
Create a conda environment using the environment.yml file with the following command:
```
conda env create -f environment.yml
```

Run the training using the following command:
```
python train.py
```

It will display the losses on the training and validation sets for 30 epochs and the test loss.
(It can take a minute to load in the torch and pandas packages)
