# DARPA-Pneumothorax

## Environment Setup

Create a conda enviornment using the requirements.txt file to install all the depended packages

`conda env create --file environment.yml`
`conda activate ai`

To update environment.yml run
`conda env export > environment.yml`



[V0 old] Create a conda enviornment using the requirements.txt file to install all the depended packages

`conda create --name <envname> --file requirements.txt`

For PackagesNotFoundError error, try

`conda config --append channels conda-forge`

To update requirements.txt run

`conda list -e > requirements.txt` 


## Dataset Setup

Download the latest version of the dataset from the CMU Box folder ("DARPA POCUS AI/Processed Datasets"). Then update the config.py or exp.yaml file to point to the dataset:

`_C.EXPERIMENT.DATASET = "DARPA-Dataset"`
`_C.DATA.ROOT_PATH = "/data1/datasets/"`

## Run

For Training or Evaluation run the main.py file and supply a exp.yaml config file with the required arguments. 

`python3 code/main.py --cfg code/configs/exp.yaml`

## Weights and Biases (wandb)

Run the below command to setup wandb and follow instruction
`wandb login`

## Error Handling

Pytorch Hub I3D model
ImportError: No module named 'fvcore' 
Ref: https://github.com/facebookresearch/detectron2/issues/11

`pip install git+https://github.com/facebookresearch/fvcore.git`