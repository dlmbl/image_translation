#!/usr/bin/env bash

# Save current directory
START_DIR=$(pwd)

# Create and navigate to GAN_code directory
mkdir -p ./GAN_code/
cd ./GAN_code/

# Clone the Git repository
git clone git@github.com:Tonks684/GANs_MI2I.git

# Create conda environment from yml
cd ./GANs_MI2I/
conda env create -f ./pix2pixHDCUDA11_environment.yml

# Activate the environment (fixed typo and deprecated usage)
source activate pix2pixHD_CUDA11

# Install additional packages
conda install -y ipykernel nbformat nbconvert black jupytext ipywidgets --name pix2pixHD_CUDA11

# Return to the previous directory
cd ../../

# Define and create the output directory (fixed variable assignment and usage)
output_dir="./data/04_image_translation/tiff_files/"
mkdir -p $output_dir

# Download and split the dataset
cd ./GAN_code/GANs_MI2I/
python download_and_split_dataset.py --output_image_folder $output_dir --crop_size 512

# Return to the starting directory
cd $START_DIR

