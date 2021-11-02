#!bin/bash

# Download the data from Google Drive to the current directory:
echo -e "download and unpack data:\n"
wget --load-cookies /tmp/cookies.txt \
  "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies \
  /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
  'https://docs.google.com/uc?export=download&id=1NSqbC46ftWNdy5CqSeRsQYkAifk3viPe' -O- | \
  sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NSqbC46ftWNdy5CqSeRsQYkAifk3viPe" \
  -O data.tar.gz && rm -rf /tmp/cookies.txt
tar -xzvf data.tar.gz

# Clone `microDL` repository, and checkout the dl_mbl_2021 branch.

echo -e "setup the microDL repo:\n"
git clone https://github.com/czbiohub/microDL.git
cd microDL
git checkout dl_mbl_2021

# create conda environment and add the module to python path.

echo -e "setup the environment:\n"
conda env create --file=conda_environment.yml
export PYTHONPATH=$PYTHONPATH:$(pwd)

# change back to the parent directory.
cd ..

