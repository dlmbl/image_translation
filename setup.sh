#!/usr/bin/env -S bash -i

# Create mamba environment
mamba create --name 04_image_translation python=3.10

# Install ipykernel in the environment.
mamba install -y ipykernel --name 04_image_translation
# Specifying the environment explicitly.
# mamba activate sometimes doesn't work from within shell scripts.


# Find path to the environment - mamba activate doesn't work from within shell scripts.
ENV_PATH=$(conda info --envs | grep 04_image_translation | awk '{print $2}')
# install viscy and its dependencies in the environment using pip.
$ENV_PATH/bin/pip install "viscy[metrics] @ git+https://github.com/mehta-lab/viscy.git"


# Store the code directory path.
CODE_DIR=$(pwd)


# Create data directory
mkdir -p ~/data/04_image_translation
cd ~/data/04_image_translation

# # Download and unzip the data
# wget https://data.broadinstitute.org/bbbc/BBBC048/BBBC048v1.zip
# unzip BBBC048v1.zip
touch here_lives_the_data

# Go back to the code directory
cd $CODE_DIR


# this didn't not work from within shell scripts on TA1 node even after mamba init.
# mamba activate 04_image_translation 