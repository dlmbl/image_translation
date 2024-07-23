#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Save current directory
START_DIR=$(pwd)
echo "Starting directory: $START_DIR"

# Create and navigate to GAN_code directory
mkdir -p ./GAN_code
cd ./GAN_code
echo "Current directory after creating and navigating to GAN_code: $(pwd)"

# Clone the Git repository
git clone git@github.com:Tonks684/GANs_MI2I.git
echo "Repository cloned. Current directory: $(pwd)"

# Create conda environment from yml
cd ./GANs_MI2I
echo "Current directory after navigating to GANs_MI2I: $(pwd)"
conda env create -f ./04_image_translation_phd.yml
echo "Conda environment created."

# Activate the environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate 04_image_translation_phd
echo "Conda environment activated."

# Install additional packages
conda install -y ipykernel nbformat nbconvert black jupytext ipywidgets
echo "Additional packages installed."

# Define and create the output directory
output_dir="../../data/04_image_translation/tiff_files"
mkdir -p "$output_dir"
echo "Output directory created at: $output_dir"

# Download and split the dataset
python download_and_split_dataset.py --output_image_folder "$output_dir" --crop_size 512
echo "Dataset downloaded and split."

# Return to the starting directory
cd "$START_DIR"
echo "Returned to the starting directory: $START_DIR"

