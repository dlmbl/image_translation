#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e
# Save current directory.
START_DIR=$(pwd)
echo "Starting directory: $START_DIR"
# Create and navigate to GAN_code directory
mkdir -p ~/data/06_image_translation/part2/GAN_code
cd ~/data/06_image_translation/part2/GAN_code
echo "Current directory after creating and navigating to GAN_code: $(pwd)"
# Clone the Git repository
git clone git@github.com:Tonks684/GANs_MI2I.git
echo "Repository cloned. Current directory: $(pwd)"
cd ~/data/06_image_translation/part2/GAN_code/GANs_MI2I
echo "Current directory after navigating to GANs_MI2I: $(pwd)"
# # Find path to the mamba environment.
ENV_PATH=$(conda info --envs | grep 06_image_translation | awk '{print $NF}')
$ENV_PATH/bin/pip install "dominate"
$ENV_PATH/bin/pip install "cellpose"
$ENV_PATH/bin/pip uninstall -y "opencv-python"
$ENV_PATH/bin/pip install "opencv-python==4.7.0.72"
$ENV_PATH/bin/pip install "ipykernel"
$ENV_PATH/bin/pip install "pandas"
$ENV_PATH/bin/pip install "pillow"
$ENV_PATH/bin/pip install "matplotlib"
$ENV_PATH/bin/pip install -U "scikit-image"
$ENV_PATH/bin/pip install "tensorboard"
$ENV_PATH/bin/pip install "viscy"
$ENV_PATH/bin/pip install "torchmetrics[detection]"
# Download the weights and pretrained tensorboards
mkdir -p ~/data/06_image_translation/part2/model_weights
mkdir -p ~/data/06_image_translation/part2/model_tensorboard
# Download into model_tensorboard and then move weights folder
cd ~/data/06_image_translation/part2/model_tensorboard
echo "Current Directory: $(pwd)"
wget -O dlmbl_requisites.zip "https://zenodo.org/record/13173900/files/dlmbl_requisites.zip?download=1"
unzip dlmbl_requisites.zip
mv ~/data/06_image_translation/part2/model_tensorboard/dlmbl_vsnuclei/dlmbl_vsnuclei ~/data/06_image_translation/part2/model_weights
mv ~/data/06_image_translation/part2/model_tensorboard/dlmbl_vscyto/dlmbl_vscyto ~/data/06_image_translation/part2/model_weights
# Download and split the dataset
cd ~/data/06_image_translation/part2/GAN_code/GANs_MI2I
echo "Curent Directory: $(pwd)"
# Define and create the output directory
output_dir="/home/smt29021/data/06_image_translation/part2/tiff_files"
mkdir -p "$output_dir"
echo "Output directory created at: $output_dir"
python download_and_split_dataset.py --output_image_folder "$output_dir" --crop_size 512
echo "Dataset downloaded and split."
# Return to the starting directory
cd "$START_DIR"
echo "Returned to the starting directory: $START_DIR"
