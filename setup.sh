#!/usr/bin/env -S bash -i

START_DIR=$(pwd)

# Create mamba environment
conda create -y --name 04_image_translation python=3.8

# Install ipykernel in the environment.
conda install -y ipykernel nbformat nbconvert black jupytext ipywidgets --name 04_image_translation
# Specifying the environment explicitly.
# mamba activate sometimes doesn't work from within shell scripts.

# install viscy and its dependencies`s in the environment using pip.
mkdir -p ~/GAN_code/
git clone https://github.com/Tonks684/GANs_MI2I.git

# Find path to the environment - mamba activate doesn't work from within shell scripts.
ENV_PATH=$(conda info --envs | grep 04_image_translation | awk '{print $NF}')
$ENV_PATH/bin/pip install ."[metrics]"

# Create data directory
output_dir = "~/data/04_image_translation/tiff_files/"
mkdir -p output_dir
# Download Data
cd "~/GAN_code/GANs_MI2I/"
python download_and_split_dataset.py --output_image_folder output_dir --crop_size 512

# Change back to the starting directory
cd $START_DIR
