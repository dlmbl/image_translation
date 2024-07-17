#!/usr/bin/env -S bash -i
START_DIR=$(pwd)
mkdir -p ./GAN_code/
cd ./GAN_code/
git clone git@github.com:Tonks684/GANs_MI2I.git
# Create conda environment from yml
cd ./GANs_MI2I/
conda env create -f ./pix2pixHD_CUDA11_environment.yml
source activate pixpi2xHD_CUDA11
# Install ipykernel in the environment.
conda install -y ipykernel nbformat nbconvert black jupytext ipywidgets --name pix2pixHD_CUDA11
cd ../../
# Create data directory
output_dir = "./data/04_image_translation/tiff_files/"
mkdir -p output_dir
# Download Data
cd ./GAN_code/GANs_MI2I/
python download_and_split_dataset.py --output_image_folder output_dir --crop_size 512

# Change back to the starting directory
cd $START_DIR
