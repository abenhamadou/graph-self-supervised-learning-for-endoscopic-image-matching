<img src="http://www.crns.rnrt.tn/front/img/logo.svg">

# Welcome to the official implementation of "Graph Self-Supervised Learning for Endoscopic Image Matching"
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![demo_vid](assets/matching_demo.gif)
# Pre-trained models 
Pre-trained models can be downloaded following the link. You may save the models to ./models folder
- [Download link](https://drive.google.com/drive/folders/1L_cCyr7Zaq4xMZ1hhwdaAFqt_GKGw45-)
# Training and testing datasets for endoscopic images matching
- [Download link](https://drive.google.com/drive/folders/1g1YD4tfo0XLffD_7lTYWldxbvR_Fu5H_)

unzip the two zip files in the ./data folder and then update your configuration yaml files accordingly (see below)
  
# Runners and configuration files
- "**run_trainning.py**", configuration yaml file in "**config/config_train.yaml**"
- "**run_validation.py**", configuration yaml file in "**config/config_validation.yaml**"
- "**run_generate_training_data.py**", configuration yaml file in "**config/config_generate_training_data.yaml**"
- "**run_matching_demo.py**", configuration yaml file in "**config/config_validation.yaml**"



## Setup for Dev on local machine
This code base is tested only on Ubuntu 20.04 LTS, TitanV and RTX2080-ti NVIDIA GPUs.
- Install local environment and requirements
First install Anaconda3 then install the requirements as follows:

> **conda create -n crns---self-sup-image-matching python=3.8**

- a new virtual environment is now created in **~/anaconda3/envs/crns---self-sup-image-matching**
Now activate the virtual environment by running:

> **source activate crns---self-sup-image-matching**

- In case you would like stop your venv **`conda deactivate`**

- To install dependencies, cd to the directory where requirements.txt is located and run the following command in your shell:

> **cat requirements.txt  | xargs -n 1 -L 1 pip3 install**

> Install **`docker`** and **`nvidia-docker`** and then run these commands:
>
> **`nvidia-docker build -t <YOUR_DOCKER_IMAGE_NAME>:<YOUR_IMAGE_VERSION> . `**
>
> **`sudo NV_GPU='0' nvidia-docker run  -i -t --entrypoint=/bin/bash --runtime=nvidia <YOUR_DOCKER_IMAGE_NAME>:<YOUR_IMAGE_VERSION>`**
>
- How to use Facial Process on local machine:
> To test photos from local folder: run this command **`python local_processing.py`**
> To process photos coming from the front on your machine: run this command **`python main.py`**


## Git pre-commit hooks
> if not already installed from the requirements.txt then first install pre-commit and black using these commands: **`pip3 install pre-commit`**
> and **`pip3 install black`**

> run **`pre-commit install`** to set up the git hook scripts
>
> You can also **`flake8 <YOURSCRIPT>.py`** to check if your python script is compliant with the project
>
> or directly fix your script using **`black <YOURSCRIPT>.py`**


## Docker build
- [comming soon]

## Known issues
