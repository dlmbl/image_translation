#!bin/bash

# Clone `microDL` repository, and checkout the dl_mbl_2021 branch.

echo -e "setup the microDL repo:\n"
git clone https://github.com/mehta-lab/microDL.git
cd microDL
git fetch --all --tags
git checkout tags/v1.0.0-rc2

# add microDL to pythonpath
export PYTHONPATH=$PYTHONPATH:$(pwd)

# change back to the parent directory.
cd ..

