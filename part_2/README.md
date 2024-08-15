# Exercise 6: Image translation - Part 2

This demo script was developed for the DL@MBL 2024 course by Samuel Tonks, Krull Lab University of Birmingham UK, with many inputs and bugfixes from [Eduardo Hirata-Miyasaki](https://github.com/edyoshikun), [Ziwen Liu](https://github.com/ziw-liu) and [Shalin Mehta](https://github.com/mattersoflight) of  CZ Biohub San Francisco.

## Image translation (Virtual Staining) via Generative Modelling

In this part of the exercise, we will tackle the same supervised image-to-image translation task but use an alternative approach. Here we will explore a generative modelling approach, specifically a conditional Generative Adversarial Network (cGAN).
<br>
<br>
The previous regression-based method learns a deterministic mapping from phase contrast to fluorescence. This results in a single virtual staining prediction to the image translation task which often leads to blurry results. Virtual staining is an ill-posed problem; given the phase contrast image, with inherent noise and lack of contrast between the background and the structure of interest, it can be very challenging to virtually stain from the phase contrast image alone. In fact, there is a distribution of possible virtual staining solutions that could come from the phase contrast.
<br>
<br>
cGANs learn to map from the phase contrast domain to this distirbution of virtual staining solutions. This distribution can then be sampled from to produce virtual staining predictions that are no longer a compromise between possible solutions which can lead to improved sharpness and realism in the generated images. Despite these improvements, cGANs can be prown to 'hallucinations' in which the network does not make a compromise when it does not know something (such as a fine-grain detail of the nuclei shape) it instead makes something up that looks very sharp and realistic. These hallucinations can appear very plausible, but in many cases to predict such details from the phase contrast is extremely challenging. This is why determining reliable evaluation criteria for the task at hand is very important when dealing with cGANs .
<br>
<br>
At a high-level a cGAN has two networks; a generator and a discriminator. The generator is a fully convolutional network that takes the source image as input and outputs the target image. The discriminator is also a fully convolutional network that takes as input the source image concatentated with a real or fake image and outputs the probabilities of whether the real fluorescence image is real or whether the fake virtual stain image is fake as shown in the figure above.<br>
<br>
The generator is trained to fool the discriminator into predicting a high probability that its generated outputs are real, and the discriminator is trained to distinguish between real and fake images. Both networks are trained using an adversarial loss in a min-max game, where the generator tries to minimize the probability of the discriminator correctly classifying its outputs as fake, and the discriminator tries to maximize this probability. It is typically trained until the discriminator can no longer determine whether or not the generated images are real or fake better than a random guess (p(0.5)).
<br>
<br>
![Overview of cGAN](https://github.com/Tonks684/dlmbl_material/blob/main/imgs/GAN.jpg?raw=true)
<br>
<br>
We will be exploring [Pix2PixHD GAN](https://arxiv.org/abs/1711.11585) architecture, a high-resolution extension of a traditional cGAN adapted for our recent [virtual staining works](https://ieeexplore.ieee.org/abstract/document/10230501?casa_token=NEyrUDqvFfIAAAAA:tklGisf9BEKWVjoZ6pgryKvLbF6JyurOu5Jrgoia1QQLpAMdCSlP9gMa02f3w37PvVjdiWCvFhA). Pix2PixHD GAN improves upon the traditional cGAN by using a coarse-to-fine generator, a multi-scale discrimator and additional loss terms. The "coarse-to-fine" generator is composed of two sub-networks, both ResNet architectures that operate at different scales. As shown below the first sub-network (G1) generates a low-resolution image, which is then upsampled and concatenated with the source image to produce a higher resolution image. The multi-scale discriminator is composed of 3 networks that operate at different scales, each network is trained to distinguish between real and fake images at that scale using the same convolution kernel size. This leads to the convolution having a much wider field of view when the inputs are downsampled. The generator is trained to fool the discriminator at each scale. 
<br>
<br>
![Pix2PixGAN ](https://github.com/Tonks684/dlmbl_material/blob/main/imgs/Pix2pixHD_1.jpg?raw=true)
<br>
<br>
The additional loss terms include a feature matching loss (as shown below), which encourages the generator to produce images that are perceptually similar to the real images at each scale. As shown below for each of the 3 discriminators, the network takes seperaetly both phase concatenated with virtual stain and phase concatenated with fluorescence stain as input and as they pass through the network the feature maps obtained for each ith layer are extracted. We then minimize the loss which is the mean L1 distance between the feature maps obtained across each of the 3 discriminators and each ith layer.
<br>
<br>
![Feature Matching Loss Pix2PixHD GAN](https://github.com/Tonks684/dlmbl_material/blob/main/imgs/Pix2pixHD_2.jpg?raw=true)
<br>
<br>
All of the discriminator and generator loss terms are weighted the same.

## Goals

As you have already explored the data in the previous parts, we will focus on training and evaluating Pix2PixHD GAN. The parts are as follows:<br>

* **Part 1** - Define dataloaders & walk through steps to train Pix2PixHD GAN.<br>
* **Part 2** - Load and assess a pre-trained Pix2PixGAN using tensorboard, discuss the different loss components and how new hyper-parameter configurations could impact performance.<br>
* **Part 3** - Evaluate performance of pre-trained Pix2PixGAN using pixel-level and instance-level metrics.<br>
* **Part 4** - Compare the performance of Viscy (regression-based) with Pix2PixHD GAN (generative modelling approach)<br>
* **Part 5** - *BONUS*: Sample different virtual staining solutions from the Pix2PixHD GAN using [MC-Dropout](https://arxiv.org/abs/1506.02142) and explore the variability and subsequent uncertainty in the virtual stain predictions.<br>


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

Activate your environment (we will use the same environment as part1)
```bash
conda activate 06_image_translation
```

## Use vscode

Install vscode, install jupyter extension inside vscode, and setup [cell mode](https://code.visualstudio.com/docs/python/jupyter-support-py). Open [solution.py](solution.py) and run the script interactively.

## Use Jupyter Notebook

The matching exercise and solution notebooks can be found [here](https://github.com/dlmbl/image_translation/) on the course repository.

Launch a jupyter environment

```
jupyter notebook
```

...and continue with the instructions in the notebook.

If `06_image_translation` is not available as a kernel in jupyter, run:

```
python -m ipykernel install --user --name=06_image_translation
```