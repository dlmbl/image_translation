#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -e
# Save current directory
START_DIR=$(pwd)
# Create and navigate to GAN_code directory
mkdir -p ./GAN_code
cd ./GAN_code
# Clone the Git repository
git clone git@github.com:Tonks684/GANs_MI2I.git
# Create conda environment from yml
cd ./GANs_MI2I
conda env create -f ./04_image_translation_phd.yml
# Activate the environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate 04_image_translation_phd
# Install additional packages
conda install -y ipykernel nbformat nbconvert black jupytext ipywidgets
# Define and create the output directory
output_dir="../../data/04_image_translation/tiff_files"
mkdir -p "$output_dir"
# Download and split the dataset
python download_and_split_dataset.py --output_image_folder "$output_dir" --crop_size 512
# Return to the starting directory
cd "$START_DIR"
