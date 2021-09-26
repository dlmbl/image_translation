#!bin/bash

# change to home directory
cd 

# copy the data to the home directory:
rsync -r --info=progress2 --info=name0 /mnt/efs/woods_hole/04_image_translation_data ~

# Download `microDL` repository, and checkout the dl_mbl_2021 branch.
git clone https://github.com/czbiohub/microDL.git
cd microDL
git checkout dl_mbl_2021

# create conda environment, activate it, and add the module to python path.
conda env create --file=conda_environment.yml
conda activate micro_dl
export PYTHONPATH=$PYTHONPATH:$(pwd)


