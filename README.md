# Exercise 4: Image translation via Generative Modelling
## Setup

Make sure that you are inside of the `image_translation` folder by using the `cd` command to change directories if needed.

Make sure that you can use conda to switch environments.

```bash
conda init
```

**Close your shell, and login again.** 

Run the setup script to create the environment for this exercise and download the dataset.
```bash
sh setup.sh
```
If you get errors relating to '\r' run
```bash
dos2unix setup.sh
```

Activate your environment
```bash
conda activate 04_image_translation
```

Launch a jupyter environment

```
jupyter notebook
```

...and continue with the instructions in the notebook.

If 04_image_translation is not available as a kernel in jupyter, run
```
python -m ipykernel install --user --name=04_image_translation
```
