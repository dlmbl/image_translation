#!bin/bash

# change to home directory
cd 

# copy the data to the home directory:
echo -e "transferring data:\n"
rsync -r --info=progress2 --info=name0 /mnt/efs/woods_hole/04_image_translation_data ~

# Download `microDL` repository, and checkout the dl_mbl_2021 branch.

echo -e "setup the microDL repo:\n"
git clone https://github.com/czbiohub/microDL.git
cd ~/microDL
git checkout dl_mbl_2021

# create conda environment and add the module to python path.

echo -e "setup the environment:\n"
conda env create --file=conda_environment.yml
export PYTHONPATH=$PYTHONPATH:$(pwd)

# change back to home directory.
cd

