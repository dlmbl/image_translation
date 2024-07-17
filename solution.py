# %% [markdown tag= "pix2pixGAN_explainer"]
"""
# Generative Modelling Approaches to Image translation
---

Written by Samuel Tonks, Krull Lab, University of Birmingham, UK.

In this part of the exercise, we will approach the same supervised image-to-image translation task as in the previous parts, but using a different model architecture. Here we will explore a generative modelling approach; a conditional Generative Adversarial Network (cGAN). 
In contrast to formulating the task as a regression problem where the model produces a single deterministic output, cGANs learn to map from the source domain to a target domain distribution. This learnt distribution can then be sampled from to produce virtual staining predictions that are no longer a compromise between possible solutions which leads to improved sharpness and realism in the generated images.

At a high-level a cGAN has two networks; a generator and a discriminator. The generator is a fully convolutional network that takes the source image as input and outputs the target image. The discriminator is also a fully convolutional network that takes as input the source image concatentated with a real or fake image and outputs the probabilities of whether the image is real or fake as shown in the Figure below: 
[View PDF](https://github.com/Tonks684/image_translation/tree/main/imgs/GAN.pdf)
The generator is trained to fool the discriminator into predicting a high probability that its generated outputs are real, and the discriminator is trained to distinguish between real and fake images. Both networks are trained using an adversarial loss in a min-max game, where the generator tries to minimize the probability of the discriminator correctly classifying its outputs as fake, and the discriminator tries to maximize this probability. It is typically trained until the discriminator can no longer determine whether or not the generated images are real or fake better than a random guess (p(0.5)).

We will be exploring [Pix2PixHD GAN](https://arxiv.org/abs/1711.11585) architecture, a high-resolution extension of a traditional cGAN adapted for our recent [virtual staining works](https://ieeexplore.ieee.org/abstract/document/10230501?casa_token=NEyrUDqvFfIAAAAA:tklGisf9BEKWVjoZ6pgryKvLbF6JyurOu5Jrgoia1QQLpAMdCSlP9gMa02f3w37PvVjdiWCvFhA). 
 Pix2PixHD GAN improves upon the traditional cGAN by using a coarse-to-fine generator, a multi-scale discrimator and additional loss terms. The "coarse-to-fine" generator is composed of two sub-networks, both ResNet architectures that operate at different scales. The first sub-network (G1) generates a low-resolution image, which is then upsampled and concatenated with the source image to produce a higher resolution image. The multi-scale discriminator is composed of 3 networks that operate at different scales, each network is trained to distinguish between real and fake images at that scale. The generator is trained to fool the discriminator at each scale. The additional loss terms include a feature matching loss, which encourages the generator to produce images that are similar to the real images at each scale. 
[View PDF](https://github.com/Tonks684/image_translation/tree/main/imgs/Pix2PixHD_1.pdf)
[View PDF](https://github.com/Tonks684/image_translation/tree/main/imgs/Pix2PixHD_2.pdf)
"""


# %% [markdown]
"""
Today, we will train a 2D image translation model using a Pix2PixHD GAN. We will use the same dataset of 301 fields of view (FOVs) of Human Embryonic Kidney (HEK) cells, each FOV has 3 channels (phase, membrane, and nuclei).
"""
# %% [markdown]
"""
<div class="alert alert-warning">
This part of the exercise is organized in 3 parts. As you have already explored the data in the previous parts, we will focus on training and evaluating Pix2PixHD GAN. The parts are as follows:

* **Part 1** - Download data, define dataloaders & start training a Pix2PixHD GAN.
* **Part 2** - Load and assess pre-trained Pix2PixGAN using tensorboard, discuss new hyper-parameter configurations.
* **Part 3** - Evaluate performance of pre-trained Pix2PixGAN using pixel-level and instance-level metrics by using Cellpose to segment the nuclei and membrane channels of the fluorescence and virtual staining images.
* **Part 4** - BONUS: Sample different virtual staining solutions from the GAN using MC-Dropout.
</div>
"""
# %% [markdown]
"""
📖 As you work through parts 2 and 3, please share the default hyper-parameter settings and the performance with everyone via [this google doc](https://docs.google.com/document/d/1hZWSVRvt9KJEdYu7ib-vFBqAVQRYL8cWaP_vFznu7D8/edit#heading=h.n5u485pmzv2z) 📖.


Our guesstimate is that each of the three parts will take ~1.5 hours. A reasonable Pix2PixHD GAN can be trained in ~1.5 hours on a typical AWS node, this notebook is designed to walk you through the training steps but load a pre-trained model and tensorboard session to ensure we can complete the exercise in the time allocated.

We will discuss your observations on google doc after checkpoints 2 and 3.

The focus of this part of the exercise is on understanding a generative modelling approach to image translation, how to train and evaluate a cGAN, and explore some hyperparameters of the cGAN. 
"""
# %% [markdown]
"""
<div class="alert alert-danger">
Set your python kernel to <span style="color:black;">04_image_translation</span>
</div>
"""
# %% <a [markdown] id="1_phase2fluor"></a>
"""
# Part 1: Download data, define dataloaders & understand how to train a model.
---------

Learning goals:

- Load dataset and configure dataloader.
- Configure Pix2PixHD GAN and train to predict nuclei from phase.
"""

# %% Imports and paths
from pathlib import Path
import os
import torch
import numpy as np
import pandas as pd
# Import all the necessary hyperparameters and configurations for training.
from GANs_MI2I.pix2pixHD.options.train_options import TrainOptions
from GANs_MI2I.pix2pixHD.options.test_options import TestOptions
# Import Pytorch dataloader and transforms.
from GANs_MI2I.pix2pixHD.data.data_loader_dlmbl import CreateDataLoader
# Import the model architecture.
from GANs_MI2I.pix2pixHD.models import create_model
# Import helper functions for visualization and processing.
from GANs_MI2I.pix2pixHD.util.visualizer import Visualizer
from GANs_MI2I.pix2pixHD.util import util
# Import train script.
from GANs_MI2I.pix2pixHD.train_dlmbl import train as train_model
from GANs_MI2I.pix2pixHD.test_dlmbl import inference as inference_model
# pytorch lightning wrapper for Tensorboard.
from torch.utils.tensorboard import SummaryWriter

# %% Imports and paths
# Initialize the default options and parse the arguments.
opt = TrainOptions().parse()
# Set the seed for reproducibility.
util.set_seed(int(opt.seed))  
# Set the experiment folder name.
opt.name = "dlmbl_vsnuclei" 
# Path to store all the logs.
opt.checkpoints_dir = Path(f"~/data/04_image_translation/{opt.name}/logs/").expanduser()
# %% [markdown]
"""
## Download Dataset (I hope wont be needed but added it to enable testing).

The same dataset as in the previous parts is used here. There should be 301 FOVs in the dataset (12 GB compressed). Each FOV consists of 3 channels of 2048x2048 images,
saved in the <a href="https://ngff.openmicroscopy.org/latest/#hcs-layout">
High-Content Screening (HCS) layout</a>
specified by the Open Microscopy Environment Next Generation File Format
(OME-NGFF).

Here we complete the following steps:

- Set the path to download data to for output_image_folder
- Download the datatset in zarr format
- For phase, nuclei and cyto channels we extract 512x512 patches from the images
- We then split the dataset into training (0.8) and validation (0.2) sets saving each image as a .tiff file., in train, validation folders respectively. 

"""
# Path to save downloaded data too.
output_image_folder = Path(
    Path("/ADD/LOCATION/TO/DOWNLOAD/DATA/TOO")
).expanduser()
# Download the dataset and split it into training and validation sets.
!python GANs_MI2I/download_and_split_dataset.py --output_image_folder {output_image_folder} --crop_size 512

# %% [markdowntags=[dataloading]]
"""
## Load Dataset & Configure Dataloaders.
Having downloaded and split our training and validation sets we now need to load the data into the model. We will use the Pytorch DataLoader class to load the data in batches. The DataLoader class is an iterator that provides a consistent way to load data in batches. We will also use the CreateDataLoader class to load the data in the correct format for the model.
"""
# %%
# Initialize the Dataset and Dataloaders.

## Define Dataset & Dataloader options.
dataset_opt = {}
dataset_opt["--dataroot"] = output_image_folder
dataset_opt["--data_type"] = "16" # Data type of the images.
dataset_opt["--loadSize"] = "512" # Size of the loaded phase image.
dataset_opt["--target"] = "nuclei" # or "cyto" depending on your choice of target for virtual stain.
dataset_opt["--input_nc"] = "1" # Number of input channels.
dataset_opt["--output_nc"] = "1" # Number of output channels.
dataset_opt["--resize_or_crop"] = "none" # Scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none].

# Update opt with key value pairs from dataset_opt.
opt.__dict__.update(dataset_opt)

# Load Training Set for input into model
train_dataloader = CreateDataLoader(opt)
dataset_train = train_dataloader.load_data()
print(f"Total Training Images = {len(train_dataloader)}")

# Load Val Set
opt.phase = "val"
val_dataloader = CreateDataLoader(opt)
dataset_val = val_dataloader.load_data()
print(f"Total Validation Images = {len(val_dataloader)}")

writer = SummaryWriter(log_dir=f"{opt.checkpoints_dir}/view_batch")

# %% [markdown]
"""
## Configure Pix2PixHD GAN and train to predict nuclei from phase.
Having loaded the data into the model we can now train the Pix2PixHD GAN to predict nuclei from phase. We will use the following hyperparameters to train the model:

"""
# %%
model_opt = {}

# Define the parameters for the Generator.
model_opt["--ngf"] = "64" # Number of filters in the generator.
model_opt["--n_downsample_global"] = "4" # Number of downsampling layers in the generator.
model_opt["--n_blocks_global"] = "9" # Number of residual blocks in the generator.
model_opt["--n_blocks_local"] = "3" # Number of residual blocks in the generator.
model_opt["--n_local_enhancers"] = "1" # Number of local enhancers in the generator.

# Define the parameters for the Discriminators.
model_opt["--num_D"] = "3" # Number of discriminators.
model_opt["--n_layers_D"] = "3" # Number of layers in the discriminator.
model_opt["--ndf"] = "32" # Number of filters in the discriminator.

# Define general training parameters.
model_opt["--gpu_ids"] = "0" # GPU ids to use. 
model_opt["--norm"] = "instance" # Normalization layer in the generator.
model_opt["--use_dropout"] = "" # Use dropout in the generator (fixed at 0.2).
model_opt["--batchSize"] = "8" # Batch size.

# Update opt with key value pairs from model_opt
opt.__dict__.update(model_opt)

# Initialize the model
phase2nuclei_model = create_model(opt)
# Define Optimizers for G and D
optimizer_G, optimizer_D = phase2nuclei_model.module.optimizer_G, phase2nuclei_model.module.optimizer_D
# Create a visualizer to perform image processing and visualization
visualizer = Visualizer(opt)


#Here will first start training a model from scrach however we can continue to train from a previously trained model by setting the following parameters.
opt.continue_train = False
if opt.continue_train:
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, "iter.txt")
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=",", dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print("Resuming from epoch %d at iteration %d" % (start_epoch, epoch_iter))
else:
     start_epoch, epoch_iter = 1, 0

# Define helper values for training
total_steps = (start_epoch-1) * (len(train_dataloader)+len(val_dataloader)) + epoch_iter 
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

train_model(opt, phase2nuclei_model, visualizer, dataset_train, dataset_val, optimizer_G, optimizer_D, total_steps, start_epoch, epoch_iter, iter_path, display_delta, print_delta, save_delta, writer)

# %% [markdown]
"""

## A heads up of what to expect from the training (more detail about this in the following section)...

The train_model function has been designed so you can see the different Pix2PixHD GAN loss components discussed in the first part of the exercise as well as additional performance measurements. As previously mentioned, Pix2PixHD GAN has two networks; a generator and a discriminator. The generator is trained to fool the discriminator into predicting a high probability that its generated outputs are real, and the discriminator is trained to distinguish between real and fake images. Both networks are trained using an adversarial loss in a min-max game, where the generator tries to minimize the probability of the discriminator correctly classifying its outputs as fake, and the discriminator tries to maximize this probability. It is typically trained until the discriminator can no longer determine whether or not the generated images are real or fake better than a random guess (p(0.5)). After a we have iterated through all the training data, we validate the performance of the network on the validation dataset. 

In light of this, we plot the discriminator probabilities of real (D_real) and fake (D_fake) images, for the training and validation datasets.

Both networks are also trained using the feature matching loss (Generator_GAN_Loss_Feat), which encourages the generator to produce images that contain similar statistics to the real images at each scale. We also plot the feature matching L1 loss for the training and validation sets together to observe the performance and how the model is fitting the data.

In our implementation, in addition to the Pix2PixHD GAN loss components already described we stabalize the GAN training by additing an additional least-square loss term. This term stabalizes the training of the GAN by penalizing the generator for producing images that the discriminator is very confident (high probability) are fake. This loss term is added to the generator loss and is used to train the generator to produce images that are similar to the real images.
We plot the least-square loss (Generator_Loss_GAN) for the training and validation sets together to observe the performance and how the model is fitting the data.
This implementation allows for the turning on/off of the least-square loss term by setting the --no_lsgan flag to the model options. As well as the turning off of the feature matching loss term by setting the --no_ganFeat_loss flag to the model options and the turning off of the VGG loss term by setting the --no_vgg_loss flag to the model options. Something you might want to explore in the next section!

Finally, we also plot the Peak-Signal-to-Noise-Ratio (PSNR) and the Structural Similarity Index Measure (SSIM) for the training and validation sets together to observe the performance and how the model is fitting the data.

[PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio), is a widely used metric to assess the quality of the generated image compared to the target image. Formally. it measures the ratio between the maximum possible power of a signal and the power of the corrupting noise that affects the fidelity of its representation. Essentially, PSNR provides a quantitative measurement of the quality of an image after compression or other processing such as image translation. Unlike the Pearson-Coeffecient, when measuring how much the pixel values of the virtual stain deviate from the target nuceli stain the score is sensitive to changes in brightness and contrast which is required for necessary for evaluating virtual staining. PSNR values range from 0dB to upper bounds that rarely exceed 60 dB. Extremely high PSNR values (above 50 dB) typically indicate almost negligible differences between the images.


[SSIM](https://en.wikipedia.org/wiki/Structural_similarity), is a perceptual metric used to measure the similarity between two images. Unlike PSNR, which focuses on pixel-wise differences, SSIM evaluates image quality based on perceived changes in structural information, luminance, and contrast. SSIM values range from -1 to 1, where 1 indicates perfect similarity between the images. SSIM is a more robust metric than PSNR, as it takes into account the human visual system"s sensitivity to structural information and contrast. SSIM is particularly useful for evaluating the quality of image translation models, as it provides a more accurate measure of the perceptual similarity between the generated and target images.

"""


# %% [markdown]
"""
<div class="alert alert-success">

## Load Pre-trained Nuclei or Cyto model

As we were limited on time, we decided that it would be most benefitial to you so load a pre-trained Pix2PixGAN. This will allow you to explore the different loss components and understand how the model trains and performs.
</div>
"""
# %% Imports and paths tags=[]
log_dir = "/PATH/TO/PRETRAINED_MODEL/TENSORBOAD/OUTPUTS"
%reload_ext tensorboard
%tensorboard --logdir {log_dir} 

# %% <a [markdown] id="1_fluor2phase"></a>
# Add in screenshots of what to expect from to see in tensorboard here (should be able to do this tomorrow AM)
"""
# Part 2: Assess trained Pix2PixGAN using tensorboard, discuss and trial new hyper-parameter configurations.
--------------------------------------------------
Learning goals:
- Understand the loss components of Pix2PixHD GAN and how they are used to train the model.
- Evaluate the fit of the model on the train and validation datasets.

In this part, we will evaluate the performance of the pre-trained model as shown in the previous part. We will begin by looking qualitatively at the model predictions, then dive into the different loss curves, as well as the SSIM and PSNR scores achieves on the validation set. We will also train another model to see if we can improve the performance of the model.

We first copy the same model parameters as the pre-trained model.
"""
# %%
model_opt = {}

# Define the parameters for the Generator.
model_opt["--ngf"] = "64" # Number of filters in the generator.
model_opt["--n_downsample_global"] = "4" # Number of downsampling layers in the generator.
model_opt["--n_blocks_global"] = "9" # Number of residual blocks in the generator.
model_opt["--n_blocks_local"] = "3" # Number of residual blocks in the generator.
model_opt["--n_local_enhancers"] = "1" # Number of local enhancers in the generator.

# Define the parameters for the Discriminators.
model_opt["--num_D"] = "3" # Number of discriminators.
model_opt["--n_layers_D"] = "3" # Number of layers in the discriminator.
model_opt["--ndf"] = "32" # Number of filters in the discriminator.

# Define general training parameters.
model_opt["--gpu_ids"] = "0" # GPU ids to use. 
model_opt["--norm"] = "instance" # Normalization layer in the generator.
model_opt["--use_dropout"] = "" # Use dropout in the generator (fixed at 0.2).
model_opt["--batchSize"] = "8" # Batch size.

#Define loss functions.
model_opt["--no_vgg_loss"] = "" # Turn off VGG loss
model_opt["--no_ganFeat_loss"] = "" # Turn off feature matching loss
model_opt["--no_lsgan"] = "" # Turn off least square loss
# Update opt with key value pairs from model_opt
opt.__dict__.update(model_opt)

# Initialize the model
phase2nuclei_model = create_model(opt)
# Define Optimizers for G and D
optimizer_G, optimizer_D = phase2nuclei_model.module.optimizer_G, phase2nuclei_model.module.optimizer_D

# Remeber to create a new name for the model outputs
opt.name = "dlmbl_vsnuclei_v2"
opt.checkpoints_dir = Path(f"~/data/04_image_translation/{opt.name}/logs/").expanduser()

# Retrain model with new hyperparameters
train_model(opt, phase2nuclei_model, visualizer, dataset_train, dataset_val, optimizer_G, optimizer_D, total_steps, start_epoch, epoch_iter, iter_path, display_delta, print_delta, save_delta, writer)

# %% [markdown]
"""
## Qualitative evaluation:

We have visualised the model output for an unseen phase contrast image and the target, nuclei stain.

- What do you notice about the virtual staining predictions? Are they realistic? How does the sharpness and visual representation compare to the regression-based approach?

- What

- What do you notice about the translation of the background pixels compared the translation of the instance pixels?

## Quantitative evaluation:

- What do you notice about the probabilities (real vs fake) of the discriminators? How do the values compare during training compared to validation?

- What do you notice about the feature matching L1 loss?

- What do you notice about the least-square loss?

- What do you notice about the PSNR and SSIM scores? Are we over or underfitting at all?

## Hyperparameter tuning:

We will train another model to see if we can improve the performance of the model. We will try to increase the number of filters in the generator and discriminator, and decrease the learning rate. We will also try to turn off the least square loss term and see how it affects the performance of the model.

- Do you notice any changes? 
- How do the different loss components change with the new hyperparameters? 
- How does the model performance change with the new hyperparameters?

"""

## View the training progress of hyper-parameter changes in tensorboard.
# %% Imports and paths tags=[]
%reload_ext tensorboard
%tensorboard --logdir {opt.checkpoints_dir} 

# %% [markdown]
"""
# Part 3: Evaluate performance of the virtual staining.
--------------------------------------------------
## Evaluate the performance of the model.
We now look at the same metrics of performance of the previous model. We typically evaluate the model performance on a held out test data. 

We will first load the test data using the same format as the training and validation data. We will then use the model to predict the nuclei channel from the phase image. We will then evaluate the performance of the model using the following metrics:

Pixel-level metrics:
- [Peak-Signal-to-Noise-Ratio (PSNR)](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio).
- [Structural Similarity Index Measure (SSIM)](https://en.wikipedia.org/wiki/Structural_similarity).

Instance-level metrics:
- [F1 score](https://en.wikipedia.org/wiki/F1_score). via [Cellpose](https://cellpose.org/).
"""

# %% Compute metrics directly and plot here.
opt = TestOptions().parse(save=False)
test_data_path = output_image_folder

opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

inference_opt = {}
    
# Ensure the parameters below align with trained model
opt.__dict__.update(model_opt)
opt.__dict__.update(dataset_opt)

# Additional Inference parameters
inference_opt["--how_many"] = "144"
inference_opt["--checkpoints_dir"] "/PATH/TO/PRETRAINED_MODEL/"
inference_opt["--results_dir"] = f"~/data/04_image_translation/{opt.name}/"
inference_opt["--which_epoch"] = "latest" # or specify the epoch number "40"
inference_opt["--phase"] = "test"
opt.__dict__.update(inference_opt)

Path(opt.results_dir).mkdir(parents=True, exist_ok=True)
test_data_loader = CreateDataLoader(opt)
test_dataset = test_data_loader.load_data()
visualizer = Visualizer(opt)

#Load pre-trained model
model = create_model(opt)

# Generate & save predictions in the results directory.
inference_model(test_dataset, opt, model)


test_metrics = pd.DataFrame(
    columns=["psnr_nuc", "SSIM_nuc"]
)

# %% Compute metrics directly and plot here.
for i, sample in enumerate(test_data.test_dataloader()):
    phase_image = sample["source"]
    with torch.inference_mode():  # turn off gradient computation.
        predicted_image = phase2fluor_model(phase_image)

    target_image = (
        sample["target"].cpu().numpy().squeeze(0)
    )  # Squeezing batch dimension.
    predicted_image = predicted_image.cpu().numpy().squeeze(0)
    phase_image = phase_image.cpu().numpy().squeeze(0)
    target_mem = min_max_scale(target_image[1, 0, :, :])
    target_nuc = min_max_scale(target_image[0, 0, :, :])
    # slicing channel dimension, squeezing z-dimension.
    predicted_mem = min_max_scale(predicted_image[1, :, :, :].squeeze(0))
    predicted_nuc = min_max_scale(predicted_image[0, :, :, :].squeeze(0))

    # Compute SSIM and pearson correlation.
    ssim_nuc = metrics.structural_similarity(target_nuc, predicted_nuc, data_range=1)
    ssim_mem = metrics.structural_similarity(target_mem, predicted_mem, data_range=1)
    pearson_nuc = np.corrcoef(target_nuc.flatten(), predicted_nuc.flatten())[0, 1]
    pearson_mem = np.corrcoef(target_mem.flatten(), predicted_mem.flatten())[0, 1]

    test_metrics.loc[i] = {
        "pearson_nuc": pearson_nuc,
        "SSIM_nuc": ssim_nuc,
        "pearson_mem": pearson_mem,
        "SSIM_mem": ssim_mem,
    }

test_metrics.boxplot(
    column=["pearson_nuc", "SSIM_nuc", "pearson_mem", "SSIM_mem"],
    rot=30,
)


# %% [markdown] tags=[]
"""
<div class="alert alert-info">

### Task 2.2 Train fluorescence to phase contrast translation model

Instantiate a data module, model, and trainer for fluorescence to phase contrast translation. Copy over the code from previous cells and update the parameters. Give the variables and paths a different name/suffix (fluor2phase) to avoid overwriting objects used to train phase2fluor models.
</div>
"""
# %% tags=[]
##########################
######## TODO ########
##########################

fluor2phase_data = HCSDataModule(
    # Your code here (copy from above and modify as needed)
)
fluor2phase_data.setup("fit")

# Dictionary that specifies key parameters of the model.
fluor2phase_config = {
    # Your config here
}

fluor2phase_model = VSUNet(
    # Your code here (copy from above and modify as needed)
)

trainer = VSTrainer(
    # Your code here (copy from above and modify as needed)
)
trainer.fit(fluor2phase_model, datamodule=fluor2phase_data)


# Visualize the graph of fluor2phase model as image.
model_graph_fluor2phase = torchview.draw_graph(
    fluor2phase_model,
    fluor2phase_data.train_dataset[0]["source"],
    depth=2,  # adjust depth to zoom in.
    device="cpu",
)
model_graph_fluor2phase.visual_graph

# %% tags=["solution"]

##########################
######## Solution ########
##########################

# The entire training loop is contained in this cell.

fluor2phase_data = HCSDataModule(
    data_path,
    source_channel="Membrane",
    target_channel="Phase",
    z_window_size=1,
    split_ratio=0.8,
    batch_size=BATCH_SIZE,
    num_workers=8,
    architecture="2D",
    yx_patch_size=YX_PATCH_SIZE,
    augment=True,
)
fluor2phase_data.setup("fit")

# Dictionary that specifies key parameters of the model.
fluor2phase_config = {
    "architecture": "2D",
    "in_channels": 1,
    "out_channels": 1,
    "residual": True,
    "dropout": 0.1,  # dropout randomly turns off weights to avoid overfitting of the model to data.
    "task": "reg",  # reg = regression task.
    "num_filters": [24, 48, 96, 192, 384],
}

fluor2phase_model = VSUNet(
    model_config=fluor2phase_config.copy(),
    batch_size=BATCH_SIZE,
    loss_function=torch.nn.functional.mse_loss,
    schedule="WarmupCosine",
    log_num_samples=5,
    example_input_yx_shape=YX_PATCH_SIZE,
)


trainer = VSTrainer(
    accelerator="gpu",
    devices=[GPU_ID],
    max_epochs=n_epochs,
    log_every_n_steps=steps_per_epoch // 2,
    logger=TensorBoardLogger(
        save_dir=log_dir,
        # lightning trainer transparently saves logs and model checkpoints in this directory.
        name="fluor2phase",
        log_graph=True,
    ),
)
trainer.fit(fluor2phase_model, datamodule=fluor2phase_data)


# Visualize the graph of fluor2phase model as image.
model_graph_fluor2phase = torchview.draw_graph(
    fluor2phase_model,
    fluor2phase_data.train_dataset[0]["source"],
    depth=2,  # adjust depth to zoom in.
    device="cpu",
)
model_graph_fluor2phase.visual_graph

# %% [markdown] tags=[]
"""
<div class="alert alert-info">

### Task 2.3

While your model is training, let"s think about the following questions:
- What is the information content of each channel in the dataset?
- How would you use image translation models?
- What can you try to improve the performance of each model?
</div>
"""
# %%
test_data_path = Path(
    "~/data/04_image_translation/HEK_nuclei_membrane_test.zarr"
).expanduser()

test_data = HCSDataModule(
    test_data_path,
    source_channel="Nuclei", # or Membrane, depending on your choice of source
    target_channel="Phase",
    z_window_size=1,
    batch_size=1,
    num_workers=8,
    architecture="2D",
)
test_data.setup("test")

test_metrics = pd.DataFrame(
    columns=["pearson_phase", "SSIM_phase"]
)


def min_max_scale(input):
    return (input - np.min(input)) / (np.max(input) - np.min(input))


# %%
for i, sample in enumerate(test_data.test_dataloader()):
    source_image = sample["source"]
    with torch.inference_mode():  # turn off gradient computation.
        predicted_image = fluor2phase_model(source_image)

    target_image = (
        sample["target"].cpu().numpy().squeeze(0)
    )  # Squeezing batch dimension.
    predicted_image = predicted_image.cpu().numpy().squeeze(0)
    source_image = source_image.cpu().numpy().squeeze(0)
    target_phase = min_max_scale(target_image[0, 0, :, :])
    # slicing channel dimension, squeezing z-dimension.
    predicted_phase = min_max_scale(predicted_image[0, :, :, :].squeeze(0))

    # Compute SSIM and pearson correlation.
    ssim_phase = metrics.structural_similarity(target_phase, predicted_phase, data_range=1)
    pearson_phase = np.corrcoef(target_phase.flatten(), predicted_phase.flatten())[0, 1]

    test_metrics.loc[i] = {
        "pearson_phase": pearson_phase,
        "SSIM_phase": ssim_phase,
    }

test_metrics.boxplot(
    column=["pearson_phase", "SSIM_phase"],
    rot=30,
)

# %% [markdown] tags=[]
"""
<div class="alert alert-success">

## Checkpoint 2
When your model finishes training, please summarize hyperparameters and performance of your models in the [this google doc](https://docs.google.com/document/d/1hZWSVRvt9KJEdYu7ib-vFBqAVQRYL8cWaP_vFznu7D8/edit#heading=h.n5u485pmzv2z)

</div>
"""

# %% <a [markdown] id="3_tuning"></a> tags=[]
"""
# Part 3: Tune the models.
--------------------------------------------------

Learning goals: Understand how data, model capacity, and training parameters control the performance of the model. Your goal is to try to underfit or overfit the model.
"""


# %% [markdown] tags=[]
"""
<div class="alert alert-info">

### Task 3.1

- Choose a model you want to train (phase2fluor or fluor2phase).
- Set up a configuration that you think will improve the performance of the model
- Consider modifying the learning rate and see how it changes performance
- Use training loop illustrated in previous cells to train phase2fluor and fluor2phase models to prototype your own training loop.
- Add code to evaluate the model using Pearson Correlation and SSIM

As your model is training, please document hyperparameters, snapshots of predictions on validation set, and loss curves for your models in [this google doc](https://docs.google.com/document/d/1hZWSVRvt9KJEdYu7ib-vFBqAVQRYL8cWaP_vFznu7D8/edit#heading=h.n5u485pmzv2z)

</div>
"""
# %% tags=[]
##########################
######## TODO ########
##########################

tune_data = HCSDataModule(
    # Your code here (copy from above and modify as needed)
)
tune_data.setup("fit")

# Dictionary that specifies key parameters of the model.
tune_config = {
    # Your config here
}

tune_model = VSUNet(
    # Your code here (copy from above and modify as needed)
)

trainer = VSTrainer(
    # Your code here (copy from above and modify as needed)
)
trainer.fit(tune_model, datamodule=tune_data)


# Visualize the graph of fluor2phase model as image.
model_graph_tune = torchview.draw_graph(
    tune_model,
    tune_data.train_dataset[0]["source"],
    depth=2,  # adjust depth to zoom in.
    device="cpu",
)
model_graph_tune.visual_graph


# %% tags=["solution"]

##########################
######## Solution ########
##########################

phase2fluor_wider_config = {
    "architecture": "2D",
    # double the number of filters at each stage
    "num_filters": [48, 96, 192, 384, 768],
    "in_channels": 1,
    "out_channels": 2,
    "residual": True,
    "dropout": 0.1,
    "task": "reg",
}

phase2fluor_wider_model = VSUNet(
    model_config=phase2fluor_wider_config.copy(),
    batch_size=BATCH_SIZE,
    loss_function=torch.nn.functional.l1_loss,
    schedule="WarmupCosine",
    log_num_samples=5,
    example_input_yx_shape=YX_PATCH_SIZE,
)


trainer = VSTrainer(
    accelerator="gpu",
    devices=[GPU_ID],
    max_epochs=n_epochs,
    log_every_n_steps=steps_per_epoch,
    logger=TensorBoardLogger(
        save_dir=log_dir,
        name="phase2fluor",
        version="wider",
        log_graph=True,
    ),
    fast_dev_run=True,
)  # Set fast_dev_run to False to train the model.
trainer.fit(phase2fluor_wider_model, datamodule=phase2fluor_data)

# %% tags=["solution"]

##########################
######## Solution ########
##########################

phase2fluor_slow_model = VSUNet(
    model_config=phase2fluor_config.copy(),
    batch_size=BATCH_SIZE,
    loss_function=torch.nn.functional.l1_loss,
    # lower learning rate by 5 times
    lr=2e-4,
    schedule="WarmupCosine",
    log_num_samples=5,
    example_input_yx_shape=YX_PATCH_SIZE,
)

trainer = VSTrainer(
    accelerator="gpu",
    devices=[GPU_ID],
    max_epochs=n_epochs,
    log_every_n_steps=steps_per_epoch,
    logger=TensorBoardLogger(
        save_dir=log_dir,
        name="phase2fluor",
        version="low_lr",
        log_graph=True,
    ),
    fast_dev_run=True,
)
trainer.fit(phase2fluor_slow_model, datamodule=phase2fluor_data)


# %% [markdown] tags=[]
"""
<div class="alert alert-success">
    
## Checkpoint 3

Congratulations! You have trained several image translation models now!
Please document hyperparameters, snapshots of predictions on validation set, and loss curves for your models and add the final perforance in [this google doc](https://docs.google.com/document/d/1hZWSVRvt9KJEdYu7ib-vFBqAVQRYL8cWaP_vFznu7D8/edit#heading=h.n5u485pmzv2z). We"ll discuss our combined results as a group.
</div>
"""
