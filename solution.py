# %% [markdown]
"""
# A Generative Modelling Approach to Image translation
Written by Samuel Tonks, Krull Lab, University of Birmingham, UK.<br><br>

---

## Introduction to Generative Modelling
In this part of the exercise, we will tackle the same supervised image-to-image translation task but use an alternative approach. Here we will explore a generative modelling approach, specifically a conditional Generative Adversarial Network (cGAN). <br>

The previous regression-based method learns a deterministic mapping from phase contrast to fluorescence. This results in a single virtual staining prediction to the image translation task which often leads to blurry results. Virtual staining is an ill-posed problem; given the phase contrast image, with inherent noise and lack of contrast between the background and the structure of interest, it can be very challenging to virtually stain from the phase contrast image alone. In fact, there is a distribution of possible virtual staining solutions that could come from the phase contrast.

cGANs learn to map from the phase contrast domain to a distirbution of virtual staining solutions. This distribution can then be sampled from to produce virtual staining predictions that are no longer a compromise between possible solutions which can lead to improved sharpness and realism in the generated images. Despite these improvements, cGANs can be prown to 'hallucinations' in which the network instead of making a compromise when it does not know something (such as a fine-grain detail of the nuclei shape) it makes something up that looks very sharp and realistic. These hallucinations can appear very plausible, but in many cases to predict such details from the phase contrast is extremely challenging. This is why determining reliable evaluation criteria for the task at hand is very important when dealing with cGANs .<br>
<br>
<br>
![Overview of cGAN](https://github.com/Tonks684/dlmbl_material/blob/main/imgs/GAN.jpg?raw=true)
<br>
<br>

At a high-level a cGAN has two networks; a generator and a discriminator. The generator is a fully convolutional network that takes the source image as input and outputs the target image. The discriminator is also a fully convolutional network that takes as input the source image concatentated with a real or fake image and outputs the probabilities of whether the real fluorescence image is real or whether the fake virtual stain image is fake as shown in the figure above.<br>

The generator is trained to fool the discriminator into predicting a high probability that its generated outputs are real, and the discriminator is trained to distinguish between real and fake images. Both networks are trained using an adversarial loss in a min-max game, where the generator tries to minimize the probability of the discriminator correctly classifying its outputs as fake, and the discriminator tries to maximize this probability. It is typically trained until the discriminator can no longer determine whether or not the generated images are real or fake better than a random guess (p(0.5)).<br>

We will be exploring [Pix2PixHD GAN](https://arxiv.org/abs/1711.11585) architecture, a high-resolution extension of a traditional cGAN adapted for our recent [virtual staining works](https://ieeexplore.ieee.org/abstract/document/10230501?casa_token=NEyrUDqvFfIAAAAA:tklGisf9BEKWVjoZ6pgryKvLbF6JyurOu5Jrgoia1QQLpAMdCSlP9gMa02f3w37PvVjdiWCvFhA). Pix2PixHD GAN improves upon the traditional cGAN by using a coarse-to-fine generator, a multi-scale discrimator and additional loss terms. The "coarse-to-fine" generator is composed of two sub-networks, both ResNet architectures that operate at different scales. As shown below the first sub-network (G1) generates a low-resolution image, which is then upsampled and concatenated with the source image to produce a higher resolution image. The multi-scale discriminator is composed of 3 networks that operate at different scales, each network is trained to distinguish between real and fake images at that scale using the same convolution kernel size. This leads to the convolution having a much wider field of view when the inputs are downsampled. The generator is trained to fool the discriminator at each scale. 
<br>
<br>
![Pix2PixGAN ](https://github.com/Tonks684/dlmbl_material/blob/main/imgs/Pix2pixHD_1.jpg?raw=true)
<br>
<br>
The additional loss terms include a feature matching loss (as shown below), which encourages the generator to produce images that are perceptually similar to the real images at each scale. As shown below for each of the 3 discriminators, the network takes seperaetly both phase concatenated with virtual stain and phase concatenated with fluorescence stain as input and as they pass through the network the feature maps obtained for each ith layer are extracted. We then minimize the loss which is the mean L1 distance between the feature maps obtained across each of the 3 discriminators and each ith layer. <br>
![Feature Matching Loss Pix2PixHD GAN](https://github.com/Tonks684/dlmbl_material/blob/main/imgs/Pix2pixHD_2.jpg?raw=true)

All of the discriminator and generator loss terms are weighted the same.
"""

# %% [markdown]
"""
Today, we will train a 2D image translation model using the Pix2PixHD GAN. We will use the same dataset of 301 fields of view (FOVs) of Human Embryonic Kidney (HEK) cells, each FOV has 3 channels (phase, membrane, and nuclei) as used in the previous section.This implementation is designed to model a single translation task at once. <br>
"""
# %% [markdown]
"""
<div class="alert alert-warning">
This part of the exercise is organized in 5 parts.<br>

As you have already explored the data in the previous parts, we will focus on training and evaluating Pix2PixHD GAN. The parts are as follows:<br>

* **Part 1** - Define dataloaders & walk through steps to train a Pix2PixHD GAN.<br>
* **Part 2** - Load and assess a pre-trained Pix2PixGAN using tensorboard, discuss the different loss components and how new hyper-parameter configurations could impact performance.<br>
* **Part 3** - Evaluate performance of pre-trained Pix2PixGAN using pixel-level and instance-level metrics.<br>
* **Part 4** - Compare the performance of Viscy (regression-based) with Pix2PixHD GAN (generative modelling approach)<br>
* **Part 5** - *BONUS*: Sample different virtual staining solutions from the Pix2PixHD GAN using [MC-Dropout](https://arxiv.org/abs/1506.02142) and explore the variability and subsequent uncertainty in the virtual stain predictions.<br>
</div>
"""
# %% [markdown]
"""
Our guesstimate is that each of the parts will take ~1 hour. A reasonable Pix2PixHD GAN can be trained in ~3.5 hours on a typical AWS node, this notebook is designed to walk you through the training steps but load a pre-trained model and tensorboard session to ensure we can complete the exercise in the time allocated. During Part 2 or 3, you're free to train your own model using the steps we outline in part 1.<br>
"""
# %% [markdown]
"""
<div class="alert alert-danger">
Set your python kernel to <span style="color:black;">04_image_translation_phd</span>
</div>
"""
# %% <a [markdown]></a>

"""
# Part 1: Define dataloaders & walk through steps to train a Pix2PixHD GAN.
---------
The focus of this part of the exercise is on understanding a generative modelling approach to image translation, how to train and evaluate a cGAN, and explore some hyperparameters of the cGAN. 

Learning goals:

- Load dataset and configure dataloader.
- Configure Pix2PixHD GAN to train for translating from phase to nuclei.
"""
# %%
from pathlib import Path
import os
import sys

# ------- PLEASE ENSURE THIS MATCHES WHERE YOU HAVE DOWNLOADED THE DLMLBL REPO -----
parent_dir = '../../data/06_image_translation/part2'
sys.path.append(parent_dir)

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import metrics
from tifffile import imread, imsave
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import all the necessary hyperparameters and configurations for training.
from GAN_code.GANs_MI2I.pix2pixHD.options.train_options import TrainOptions
from GAN_code.GANs_MI2I.pix2pixHD.options.test_options import TestOptions

# Import Pytorch dataloader and transforms.
from GAN_code.GANs_MI2I.pix2pixHD.data.data_loader_dlmbl import CreateDataLoader

# Import the model architecture.
from GAN_code.GANs_MI2I.pix2pixHD.models import create_model

# Import helper functions for visualization and processing.
from GAN_code.GANs_MI2I.pix2pixHD.util.visualizer import Visualizer
from GAN_code.GANs_MI2I.pix2pixHD.util import util

# Import train script.
from GAN_code.GANs_MI2I.pix2pixHD.train_dlmbl import train as train_model
from GAN_code.GANs_MI2I.pix2pixHD.test_dlmbl import inference as inference_model
from GAN_code.GANs_MI2I.pix2pixHD.test_dlmbl import sampling

# Import the function to compute segmentation scores.
from GAN_code.GANs_MI2I.segmentation_scores import gen_segmentation_scores
# pytorch lightning wrapper for Tensorboard.
from torch.utils.tensorboard import SummaryWriter


# Initialize the default options and parse the arguments.
opt = TrainOptions().parse()
# Set the seed for reproducibility.
util.set_seed(42)
# Set the experiment folder name.
translation_task = "nuclei"  # or "cyto" depending on your choice of target for virtual stain.
opt.name = "dlmbl_vsnuclei"
# Path to store all the logs.
opt.checkpoints_dir = Path(f"./GAN_code/GANs_MI2I/new_training_runs/").expanduser()
Path(f'{opt.checkpoints_dir}/{opt.name}').mkdir(parents=True, exist_ok=True)
output_image_folder = Path("./data/04_image_translation/tiff_files/").expanduser()
# Initalize the tensorboard writer.
writer = SummaryWriter(log_dir=opt.checkpoints_dir)
# %% [markdown]
"""
## 1.1 Load Dataset & Configure Dataloaders.<br>
Having already downloaded and split our training, validation and test sets we now need to load the data into the model. We will use the Pytorch DataLoader class to load the data in batches. The DataLoader class is an iterator that provides a consistent way to load data in batches. We will also use the CreateDataLoader class to load the data in the correct format for the Pix2PixHD GAN.
"""
# %%
# Initialize the Dataset and Dataloaders.

## Define Dataset & Dataloader options.
opt.dataroot = output_image_folder
opt.data_type = 16  # Data type of the images.
opt.loadSize = 512  # Size of the loaded phase image.
opt.input_nc = 1  # Number of input channels.
opt.output_nc = 1  # Number of output channels.
opt.resize_or_crop = "none"  # Scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none].
opt.target = "nuclei"  # or "cyto" depending on your choice of target for virtual stain.

# Load Training Set for input into model
train_dataloader = CreateDataLoader(opt)
dataset_train = train_dataloader.load_data()
print(f"Total Training Images = {len(train_dataloader)}")

# Load Val Set
opt.phase = "val"
val_dataloader = CreateDataLoader(opt)
dataset_val = val_dataloader.load_data()
print(f"Total Validation Images = {len(val_dataloader)}")
opt.phase= "train"
# %% [markdown]
"""
## Configure Pix2PixHD GAN and train to predict nuclei from phase.
Having loaded the data into the model we can now train the Pix2PixHD GAN to predict nuclei from phase. We will use the following hyperparameters to train the model:

"""
# %%
# Define the parameters for the Generator.
opt.ngf = 64  # Number of filters in the generator.
opt.n_downsample_global = 4  # Number of downsampling layers in the generator.
opt.n_blocks_global = 9  # Number of residual blocks in the generator.
opt.n_blocks_local = 3  # Number of residual blocks in the generator.
opt.n_local_enhancers = 1  # Number of local enhancers in the generator.

# Define the parameters for the Discriminators.
opt.num_D = 3  # Number of discriminators.
opt.n_layers_D = 3  # Number of layers in the discriminator.
opt.ndf = 32  # Number of filters in the discriminator.

# Define general training parameters.
opt.gpu_ids = [0] # GPU ids to use.
opt.norm = "instance"  # Normalization layer in the generator.
opt.use_dropout = ""  # Use dropout in the generator (fixed at 0.2).
opt.batchSize = 8  # Batch size.

# Create a visualizer to perform image processing and visualization
visualizer = Visualizer(opt)


# Here will first start training a model from scrach however we can continue to train from a previously trained model by setting the following parameters.
opt.continue_train = False
if opt.continue_train:
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, "iter.txt")
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=",", dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print("Resuming from epoch %d at iteration %d" % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0
    
print('------------ Options -------------')
for k, v in sorted(vars(opt).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

# Set the number of epoch to be 1 for demonstration purposes
opt.n_epochs = 2 # start from 1
# Initialize the model
phase2nuclei_model = create_model(opt)
# Define Optimizers for G and D
optimizer_G, optimizer_D = (
    phase2nuclei_model.module.optimizer_G,
    phase2nuclei_model.module.optimizer_D,
)
# %%
train_model(
    opt,
    phase2nuclei_model,
    visualizer,
    dataset_train,
    dataset_val,
    optimizer_G,
    optimizer_D,
    start_epoch,
    epoch_iter,
    writer,
)
# %% [markdown]
"""
<div class="alert alert-info">

## A heads up of what to expect from the training...
<br>
</div>
"""
# %% [markdown]
"""
<div class="alert alert-info">

**Visualise Phase, Fluorescence and Virtual Stain for Validation Examples**<br>
- We can observe how the performance improves over time using the images tab and the sliding window.
<br><br>
</div>
"""
# %% [markdown]
"""
<div class="alert alert-info">

**Discriminator Predicted Probabilities**<br>
- We plot the discriminator's predicted probabilities that the phase with fluorescence is phase and fluorescence and that the phase with virtual stain is phase with virtual stain. It is typically trained until the discriminator can no longer classify whether or not the generated images are real or fake better than a random guess (p(0.5)). We plot this for both the training and validation datasets.<br><br>
</div>
"""
# %% [markdown]
"""
<div class="alert alert-info">

**Adversarial Loss**<br>
- We can formulate the adversarial loss as a Least Squared Error Loss in which for real data the discriminator should output a value close to 1 and for fake data a value close to 0. The generator's goal is to make the discriminator output a value as close to 1 for fake data. We plot the least squared error loss.
<br><br>
</div>
"""
# %% [markdown]
"""
<div class="alert alert-info">

**Feature Matching Loss**<br>
- Both networks are also trained using the generator feature matching loss which encourages the generator to produce images that contain similar statistics to the real images at each scale. We also plot the feature matching L1 loss for the training and validation sets together to observe the performance and how the model is fitting the data.<br><br>
</div>
"""
# %% [markdown]
"""
<div class="alert alert-info">

This implementation allows for the turning on/off of the least-square loss term by setting the opt.no_lsgan flag to the model options. As well as the turning off of the feature matching loss term by setting the opt.no_ganFeat_loss flag to the model options. Something you might want to explore in the next section!<br><br>

</div>
"""
# %% [markdown]
"""
<div class="alert alert-success">
    
## Checkpoint 1

Congratulations! You should now have a better understanding of how a conditional generative model works!

</div>
"""
# %% <a [markdown]></a>
"""
# Part 2: Load & Assess trained Pix2PixGAN using tensorboard, discuss performance of the model.
--------------------------------------------------
Learning goals:
- Load a pre-trained Pix2PixHD GAN model for either phase to nuclei or phase to cyto (lets start with phase to nuclei: dlmbl_vsnuclei)
- Discuss the loss components of Pix2PixHD GAN and how they are used to train the model.
- Evaluate the fit of the model on the train and validation datasets.

In this part, we will evaluate the performance of the pre-trained model. We will begin by looking qualitatively at the model predictions, then dive into the different loss curves, as well as the SSIM and PSNR scores achieves on the validation set. We will explore the implications of different hyper-parameter combinations for the performance of the model.

"""
# %%
log_dir = f"~/data/06_image_translation/part2/model_tensorboard/{opt.name}/"
%reload_ext tensorboard
# %%
%tensorboard --logdir $log_dir
# %% [markdown]
"""
<div class="alert alert-info">


## Qualitative evaluation:
<br>
We have visualised the model output for an unseen phase contrast image and the target, nuclei stain.<br><br>
Please note down your thoughts about the following questions...
<br><br>

1.**What do you notice about the virtual staining predictions? How do they appear compared to the regression-based approach? Can you spot any hallucinations?** 
<br>
</div>
"""
# %% [markdown]
"""
<div class="alert alert-info">



## Quantitative evaluation:

2. What do you notice about the probabilities of the discriminators? How do the values compare during training compared to validation?<br><br>
3. What do you notice about the feature matching L1 loss?<br><br>
4. What do you notice about the least-square loss?<br><br>
5. What do you notice about the PSNR and SSIM scores? Are we over or underfitting at all?**<br><br>
</div>
"""

# %% [markdown]
"""
<div class="alert alert-success">
    
## Checkpoint 2

Congratulations! You should now have a better understanding the different loss components of Pix2PixHD GAN and how they are used to train the model. You should also have a good understanding of the fit of the model during training on the training and validation datasets.

</div>
"""

# %% [markdown]
"""
# Part 3: Evaluate performance of the virtual staining on unseen data.
--------------------------------------------------
## Evaluate the performance of the model.
We now look at the same metrics of performance of the previous model. We typically evaluate the model performance on a held out test data. 

Steps:
- Define our model parameters for the pre-trained model (these are the same parameters as shown in earlier cells but copied here for clarity).
- Load the test data.

We will first load the test data using the same format as the training and validation data. We will then use the model to predict the nuclei channel from the phase image. We will then evaluate the performance of the model using the following metrics:

Pixel-level metrics:
- [Peak-Signal-to-Noise-Ratio (PSNR)](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio).
- [Structural Similarity Index Measure (SSIM)](https://en.wikipedia.org/wiki/Structural_similarity).

Instance-level metrics:
- [F1 score](https://en.wikipedia.org/wiki/F1_score). via [Cellpose](https://cellpose.org/).
"""

# %%
opt = TestOptions().parse(save=False)

# Define the parameters for the dataset.
opt.dataroot = output_image_folder
opt.data_type = 16  # Data type of the images.
opt.loadSize = 512  # Size of the loaded phase image.
opt.input_nc = 1  # Number of input channels.
opt.output_nc = 1  # Number of output channels.
opt.target = "nuclei"  # "nuclei" or "cyto" depending on your choice of target for virtual stain.
opt.resize_or_crop = "none"  # Scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none].
opt.batchSize = 1 # Batch size for training

# Define the model parameters for the pre-trained model.

# Define the parameters for the Generator.
opt.ngf = 64  # Number of filters in the generator.
opt.n_downsample_global = 4  # Number of downsampling layers in the generator.
opt.n_blocks_global = 9  # Number of residual blocks in the generator.
opt.n_blocks_local = 3  # Number of residual blocks in the generator.
opt.n_local_enhancers = 1  # Number of local enhancers in the generator.

# Define the parameters for the Discriminators.
opt.num_D = 3  # Number of discriminators.
opt.n_layers_D = 3  # Number of layers in the discriminator.
opt.ndf = 32  # Number of filters in the discriminator.

# Define general training parameters.
opt.gpu_ids= [0]  # GPU ids to use.
opt.norm = "instance"  # Normalization layer in the generator.
opt.use_dropout = ""  # Use dropout in the generator (fixed at 0.2).
opt.batchSize = 8  # Batch size.

# Define loss functions.
opt.no_vgg_loss = ""  # Turn off VGG loss
opt.no_ganFeat_loss = ""  # Turn off feature matching loss
opt.no_lsgan = ""  # Turn off least square loss

# Additional Inference parameters
opt.name = f"dlmbl_vsnuclei"
opt.how_many = 112  # Number of images to generate.
opt.checkpoints_dir = f"./GAN_code/GANs_MI2I/pre_trained/{opt.name}/"  # Path to the model checkpoints.
opt.results_dir = f"./GAN_code/GANs_MI2I/pre_trained/{opt.name}/inference_results/"  # Path to store the results.
opt.which_epoch = "latest"  # or specify the epoch number "40"
opt.phase = "test"

opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
Path(opt.results_dir).mkdir(parents=True, exist_ok=True)

# Load the test data.
test_data_loader = CreateDataLoader(opt)
test_dataset = test_data_loader.load_data()
visualizer = Visualizer(opt)

# Load pre-trained model
model = create_model(opt)

# %%
# Generate & save predictions in the results directory.
inference_model(test_dataset, opt, model)

# %%
# Gather results for evaluation
virtual_stain_paths = sorted([i for i in Path(opt.results_dir).glob("**/*.tiff")])
target_stain_paths = sorted([i for i in Path(f"{output_image_folder}/{translation_task}/test/").glob("**/*.tiff")])
phase_paths = sorted([i for i in Path(f"{output_image_folder}/input/test/").glob("**/*.tiff")])
assert (len(virtual_stain_paths) == len(target_stain_paths) == len(phase_paths)
), f"Number of images do not match. {len(virtual_stain_paths)},{len(target_stain_paths)} {len(phase_paths)} "

# Create arrays to store the images.
virtual_stains = np.zeros((len(virtual_stain_paths), 512, 512))
target_stains = virtual_stains.copy()
phase_images = virtual_stains.copy()
# Load the images and store them in the arrays.
for index, (v_path, t_path, p_path) in tqdm(
    enumerate(zip(virtual_stain_paths, target_stain_paths, phase_paths))
):
    virtual_stain = imread(v_path)
    phase_image = imread(p_path)
    target_stain = imread(t_path)
    # Append the images to the arrays.
    phase_images[index] = phase_image
    target_stains[index] = target_stain
    virtual_stains[index] = virtual_stain
    
# %% [markdown] tags=[]
"""
<div class="alert alert-info">

### Task 3.1 Visualise the results of the model on the test set.

Create a matplotlib plot that visalises random samples of the phase images, target stains, and virtual stains.
</div>
"""

# %% tags=["task"]
##########################
######## TODO ########
##########################

def visualise_results():
    # Your code here
    pass


# %% tags=["solution"]

##########################
######## Solution ########
##########################

def visualise_results(
    phase_images: np.array, target_stains: np.array, virtual_stains: np.array, crop_size=None
):
    """
    Visualizes the results of the image processing algorithm.

    Args:
        phase_images (np.array): Array of phase images.
        target_stains (np.array): Array of target stain images.
        virtual_stains (np.array): Array of virtual stain images.
        crop_size (int, optional): Size of the crop. Defaults to None.
    """

    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    sample_indices = np.random.choice(len(phase_images), 5)
    if crop_size is not None:
        phase_images = phase_images[:,:crop_size,:crop_size]
        target_stains = target_stains[:,:crop_size,:crop_size]
        virtual_stains = virtual_stains[:,:crop_size,:crop_size]
    for i, idx in enumerate(sample_indices):
        axes[i, 0].imshow(phase_images[idx], cmap="gray")
        axes[i, 0].set_title("Phase")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(
            target_stains[idx],
            cmap="gray",
            vmin=np.percentile(target_stains[idx], 1),
            vmax=np.percentile(target_stains[idx], 99),
        )
        axes[i, 1].set_title("Target Fluorescence ")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(
            virtual_stains[idx],
            cmap="gray",
            # vmin=np.percentile(target_stains[idx], 1),
            # vmax=np.percentile(target_stains[idx], 99),
        )
        axes[i, 2].set_title("Virtual Stain")
        axes[i, 2].axis("off")
    plt.tight_layout()
    plt.show()
visualise_results(phase_images, target_stains,virtual_stains)
# %% [markdown] tags=[]
"""
<div class="alert alert-info">

### Task 3.2 Compute pixel-level metrics

Compute the pixel-level metrics for the virtual stains and target stains. The metrics include Pearson correlation, SSIM, and PSNR.
</div>
"""
# %%
test_metrics = pd.DataFrame(columns=["pearson_nuc", "SSIM_nuc", "psnr_nuc"])
# Pixel-level metrics
for i, (target_image, predicted_image) in enumerate(zip(target_stains, virtual_stains)):
    # Compute SSIM and pearson correlation.
    ssim_score = metrics.structural_similarity(
        target_image, predicted_image, data_range=1
    )
    pearson_score = np.corrcoef(target_image.flatten(), predicted_image.flatten())[0, 1]
    psnr_score = metrics.peak_signal_noise_ratio(
        target_image, predicted_image, data_range=1
    )
    test_metrics.loc[i] = {
        "pearson_nuc": pearson_score,
        "SSIM_nuc": ssim_score,
        "psnr_nuc": psnr_score,
    }

test_metrics.boxplot(
    column=["pearson_nuc", "SSIM_nuc"], #,, "psnr_nuc"],
    rot=30,
)

# %% [markdown]

"""
<div class="alert alert-info">

### Task 3.3 Compute instance-level metrics

- Use Cellpose to segment the nuclei or  membrane channels of the fluorescence and virtual staining images.
- Compute the F1 score for the segmentation masks.


</div>
"""
# %%
# Run cellpose to generate masks for the virtual stains
path_to_virtual_stain = Path(opt.results_dir)
path_to_targets = Path(f"{output_image_folder}/test/")
cellpose_model = "nuclei"  # or "cyto" depending on your choice of target for virtual stain.
# %%
# Run for virtual stain
import subprocess
command = [ "python", "-m", "cellpose", "--dir", "./GAN_code/GANs_MI2I/pre_trained/dlmbl_vsnuclei/inference_results", "--pretrained_model", "nuclei","--chan", "0", "--save_tif", "--verbose"]
result = subprocess.run(command,capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
# %%
predicted_masks = sorted([i for i in path_to_virtual_stain.glob("**/*_cp_masks.tif*")])
target_masks = sorted([i for i in Path('./data/04_image_translation/tiff_files/nuclei/masks/').glob("**/*.tiff")])
print(predicted_masks[:3], target_masks[:3])
assert len(predicted_masks) == len(target_masks), f"Number of masks do not match {len(predicted_masks)}, {len(target_masks)}"
# %%


def visualise_results_and_masks(
    phase_images: np.array, target_stains: np.array, virtual_stains: np.array, target_masks_paths: list, virtual_masks_paths: list, crop_size=None
):
    """
    Visualizes the results of the image processing algorithm.

    Args:
        phase_images (np.array): Array of phase images.
        target_stains (np.array): Array of target stain images.
        virtual_stains (np.array): Array of virtual stain images.
        target_masks_paths (list): list of target stain mask paths.
        virtual_masks_paths (list): list of virtual stain mask paths.
        crop_size (int, optional): Size of the crop. Defaults to None.
    """

    fig, axes = plt.subplots(3, 5, figsize=(15, 20))
    sample_indices = np.random.choice(len(phase_images),3)
    if crop_size is not None:
        phase_images = phase_images[:,:crop_size,:crop_size]
        target_stains = target_stains[:,:crop_size,:crop_size]
        virtual_stains = virtual_stains[:,:crop_size,:crop_size]
        
    for i, idx in enumerate(sample_indices):
        axes[i, 0].imshow(phase_images[idx], cmap="gray")
        axes[i, 0].set_title("Phase")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(
            target_stains[idx],
            cmap="gray",
            vmin=np.percentile(target_stains[idx], 1),
            vmax=np.percentile(target_stains[idx], 99),
        )
        axes[i, 1].set_title("Target Fluorescence ")
        axes[i, 1].axis("off")
        target_mask = imread(target_masks_paths[idx]).astype(np.uint8)
        axes[i, 2].imshow(
            target_mask,
            cmap="inferno",)
        axes[i, 2].set_title("Target Fluorescence Mask")
        axes[i, 2].axis("off")
        axes[i, 3].imshow(
            virtual_stains[idx],
            cmap="gray",
            # vmin=np.percentile(target_stains[idx], 1),
            # vmax=np.percentile(target_stains[idx], 99),
        )
        axes[i, 3].set_title("Virtual Stain")
        axes[i, 3].axis("off")
        virtual_mask = imread(virtual_masks_paths[idx]).astype(np.uint8)       
        axes[i, 4].imshow(
            virtual_mask,
            cmap="inferno",
            # vmin=np.percentile(target_stains[idx], 1),
            # vmax=np.percentile(target_stains[idx], 99),
        )
        axes[i, 4].set_title("Virtual Stain Mask")
        axes[i, 4].axis("off")
    plt.tight_layout()
    plt.show()
    
visualise_results_and_masks(phase_images, target_stains,virtual_stains,target_masks,predicted_masks)
# %% [markdown]
# Use a predefined function to compute F1 score and its component parts.

# %%
# Generate dataframe to store the outputs
results = pd.DataFrame(
    columns=[
        'Model', 'Image', 'GT_Cell_Count','Threshold', 'F1', 'IoU',
        'TP', 'FP', 'FN', 'Precision', 'Recall'
    ],
) 
# Create inputs to function
image_sets = []
for i in range(len(predicted_masks)):
    name = str(predicted_masks[i]).split("/")[-1] 
    virtual_stain_mask = imread(predicted_masks[i])
    fluorescence_mask = imread(target_masks[i])  
    image_sets.append(
        {
            "Image": name,
            "Model": "Pix2PixHD",
            "Virtual_Stain_Mask": virtal_stain_mask,
            "Fluorescence_Mask": fluorescence_mask,
        }
    )
# Compute the segmentation scores
results, _, _ = \
    gen_segmentation_scores(
        image_sets, results, final_score_output=f"./GAN_code/GANs_MI2I/pre_trained/{opt.name}/inference_results/")

results.head()
# %%
# Get Mean F1 results
mean_f1 = results["F1"].mean()
std_f1 = results["F1"].std()
print(f"Mean F1 Score: {np.round(mean_f1,2)}")

plt.hist(results["F1"], bins=10)
plt.xlabel("F1 Score")
plt.ylabel("Frequency")
plt.title(f"F1 Score: Mu {mean_f1}+-{std_f1}")

# %% [markdown]
"""
<div class="alert alert-success">
    
## Checkpoint 3

Congratulations! You have generated predictions from a pre-trained model and evaluated the performance of the model on unseen data. You have computed pixel-level metrics and instance-level metrics to evaluate the performance of the model. You may have also began training your own Pix2PixHD GAN models with alternative hyperparameters.
Please document hyperparameters, snapshots of predictions on validation set, and loss curves for your models and add the final perforance in [this google doc](ADD LINK TO SHARED DOC). We"ll discuss our combined results as a group.
</div>
"""

# %% [markdown]
"""
# Part 4. Visualise Regression vs Generative Modelling Approaches
--------------------------------------------------
"""
# %% tags=["task"]
# Load Viscy Virtual Stains
viscy_results_path = "/ADD/PATH/TO/RESULTS/HERE"
viscy_stain_paths = sorted([i for i in Path(viscy_results_path).glob("**/*.tiff")])
assert len(viscy_stain_paths) == len(virtual_stain_paths), "Number of images do not match."
visy_stains = np.zeros((len(viscy_stain_paths), 512, 512))
for index, v_path in enumerate(viscy_stain_paths):
    viscy_stain = imread(v_path)
    visy_stains[index] = viscy_stain

##########################
######## TODO ########
##########################


def visualise_both_methods():
    # Your code here
    pass


# %% tags=["solution"]

##########################
######## Solution ########
##########################

def visualise_both_methods(
    phase_images: np.array, target_stains: np.array, pix2pixHD_results: np.array, viscy_results: np.array,crop_size=None
):
    fig, axes = plt.subplots(5, 4, figsize=(15, 15))
    sample_indices = np.random.choice(len(phase_images), 5)
    if crop is not None:
        phase_images = phase_images[:,:crop_size,:crop_size]
        target_stains = target_stains[:,:crop_size,:crop_size]
        pix2pixHD_results = pix2pixHD_results[:,:crop_size,:crop_size]
        viscy_results = viscy_results[:,:crop_size,:crop_size]

    for i, idx in enumerate(sample_indices):
        axes[i, 0].imshow(phase_images[idx], cmap="gray")
        axes[i, 0].set_title("Phase")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(
            target_stains[idx],
            cmap="gray",
            vmin=np.percentile(target_stains[idx], 1),
            vmax=np.percentile(target_stains[idx], 99),
        )
        axes[i, 1].set_title("Nuclei")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(
            viscy_results[idx],
            cmap="gray",
            vmin=np.percentile(target_stains[idx], 1),
            vmax=np.percentile(target_stains[idx], 99),
        )
        axes[i, 2].set_title("Regression\nVirtual Stain")
        axes[i, 2].axis("off")
        
        axes[i, 3].imshow(
            pix2pixHD_results[idx],
            cmap="gray",
            vmin=np.percentile(target_stains[idx], 1),
            vmax=np.percentile(target_stains[idx], 99),
        )
        axes[i, 3].set_title("Pix2PixHD GAN\nVirtual Stain")
        axes[i, 3].axis("off")
    plt.tight_layout()
    plt.show()

# %% [markdown]
"""
# Part 5: BONUS: Sample different virtual staining solutions from the GAN using MC-Dropout and explore the uncertainty in the virtual stain predictions.
--------------------------------------------------
Steps:
- Load the pre-trained model.
- Generate multiple predictions for the same input image.
- Compute the pixel-wise variance across the predictions.
- Visualise the pixel-wise variance to explore the uncertainty in the virtual stain predictions.

"""
# %%
# Use the same model and dataloaders as before.
# Load the test data.
test_data_loader = CreateDataLoader(opt)
test_dataset = test_data_loader.load_data()
visualizer = Visualizer(opt)

# Load pre-trained model
opt.variational_inf_runs = 100 # Number of samples per phase input
opt.variation_inf_path = f"./GAN_code/GANs_MI2I/pre_trained/{opt.name}/samples/"  # Path to store the samples.
opt.dropout_variation_inf = True  # Use dropout during inference.
model = create_model(opt)
# Generate & save predictions in the variation_inf_path directory.
sampling(test_dataset, opt, model)
                                      
# %%
# Visualise Samples                                      
samples = sorted([i for i in Path(f"./GAN_code/GANs_MI2I/pre_trained/{opt.name}/samples").glob("**/*mask*.tif*")])
# Create arrays to store the images.
sample_images = np.zeros((len(samples),112, 512, 512)) # (samples, images, height, width)
# Load the images and store them in the arrays.
for index, sample_path in tqdm(enumerate(samples)):
    sample_image = imread(sample_path)
    # Append the images to the arrays.
    sample_images[index] = sample_image
# Plot the phase image, the target image, the variance of samples and 3 samples

# Create a matplotlib plot with animation through images.
import matplotlib.animation as animation

def animate_images(images):
    fig, ax = plt.subplots()
    ax.axis('off')
    im = ax.imshow(images[0], cmap='gray')

    def update(i):
        im.set_array(images[i])
        return im,

    ani = animation.FuncAnimation(fig, update, frames=len(images), interval=200)
    plt.show()

animate_images(sample_images)

# Visualise the results of the model on the test set.
fig, axes = plt.subplots(3, 7, figsize=(20, 5))
sample_indices = np.random.choice(sample_images.shape[1], 3)
for row, indices in enumerate(sample_indices):
    axes[row, 0].imshow(phase_images[indices], cmap="gray")
    axes[row, 0].set_title("Phase")
    axes[row,0].axis("off")
    axes[row, 1].imshow(target_stains[indices], cmap="gray")
    axes[row, 1].set_title("Target Fluorescence")
    axes[row,1].axis("off")
    variance = np.var(sample_images[:,indices], axis=0)
    axes[row, 2].imshow(variance, cmap="inferno")
    axes[row, 2].set_title("Pixel-wise Sample Variance")
    axes[row, 2].axis("off")
    for col in range(3, 7):
        axes[row, col].imshow(sample_images[col-3,indices], cmap="gray")
        axes[row, col].set_title(f"Sample {col-3}")
        axes[row,col].axis("off")
plt.tight_layout()
plt.show()                          
    
                                      

