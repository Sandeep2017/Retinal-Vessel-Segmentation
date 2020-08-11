# Retinal Vessel Segmentation
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)


A subpixel dense and deeply supervised U-Net like architecture for segmenting retinal vessels.

## Overview
* [Contributors](#Contributors)
* [Introduction](#Introduction)
* [Dataset Used](#Dataset-And-Preprocessing)
* [Model Architecture](#Network-Architecture)
* [Loss Function & Optimizer](#Loss-Function-And-Optimizer)
* [Learning Rate](#Learning-Rate)
* [Result](#Result)
* [Citations](#Citations)

### Contributors:
This project is created by the joint efforts of
* [Subham Singh](https://github.com/Subham2901)
* [Sandeep Ghosh](https://github.com/Sandeep2017)
* [Amit Maity](https://github.com/Neel1097)

### Introduction:
Automatic extraction of retinal vessels from retinal fundas images is a long researched area in AI aided medical diagnosis as it assists in the implementation of screening programs for diabetic retinopathy, to establish a relationship between vessel tortuosity and hypertensive retinopathy, vesssel diameter measurement in relation with diagnosis of hypertension, and computer assisted laser surgery. Also, the retinal vascular tree is found to be unique for each individual and can be used for biometric identification. 
For all the above mentioned reasons and for many others, automated retinal vessel extraction is a very important task. In this project, we have tried to use a modified [U-Net](https://arxiv.org/abs/1505.04597) like structure and achieved near state-of-the-art results. 

### Dataset Used
The [DRIVE dataset](https://drive.grand-challenge.org/) is used here for training as well as testing purposes.
The dataset consists of 40 expert annotated retinal fundas images divided equally into a training and test set of 20 image each. Each image was captured using 8 bits per color plane at 768 by 584 pixels. The FOV of each image is circular with a diameter of approximately 540 pixels. For this dataset, the images have been cropped around the FOV. For each image, a mask image is provided that delineates the FOV. 
A sample fundas image along with its hand annotated mask is provided below.

Original Image             |  Mask Image               |        AV Mask
:-------------------------:|:-------------------------:|:-------------------------:|
![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/29-training.png)  |  ![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/29_training.png)   |   ![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/29_training%20(2).png)

### Augmentation & Preprocessing

The training data was augmented on the fly using the [Albumentations library](https://albumentations.ai/).
A strong combination of different types of image augmentations were applied with varied probabilities. They were:
* Random Flips
* Transpose
* Scale, Shift & rotate
* Random Rotations
* Optical Distortion
* Grid Distortion
* Elastic Transform
* RGB Shift
* Random Gamma
* Random Brightness & contrast

Along with the above mentioned augmentations, every image in the training and testing sets underwent a Histogram Equalization preprocessing step, i.e, CLAHE (Contrast Limited Adaptive Histogram Equalization).

Some examples of augmented images and masks are given below.

Augmented Training Images
:-------------------------:|
![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/x.PNG)

Augmented Training Masks
:-------------------------:|
![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/y.PNG)|

### Network Architecture
* The proposed architecture shown in Fig. 1, below is inspired from the original U-Net architecure with major modifications. It is a 4 stage segmentation architecture based upon the concept of stacking. In this 4 stage architecture, the feature map from the penultimate layer of the first stage is passed on to the second stage as the penultimate layer contains more information than the last output layer. The passed on feature map is then concatenated with the original input shape (shown in bold black arrows) and is fed as an input to the second stage. Concatenating the original shape image with the previous feature map improves the refinement of the predictions. Similarly, the feature map from the penultimate layer of the second stage is passed on the third stage and concatenated with the original input image as so on until the fourth stage. The final predictions are obtained from the output of the fourth stage.

* Here the learnable transposed convolution method or the interpolation based up-sampling method is replaced by the efficient [subpixel convolutions](xxxxx).
A subpixel convolutional layer is shown in Fig. 2, which is just a standard 1x1 convolution layer followed by a pixel shuffling operation which re-arranges the pixels from the depth dimensions to the spatial dimensions. This type pf convolutions minimize information loss during up-sampling of the image in the decoder network.

* Each stage has its own output layer and the loss is minimized at each stage. The architecture is trained using deep supervision.

* The parallel layers of the four decoder networks are connected via dense connections which helps in improving the spatial knowledge transfer through each stage of the model while training.


![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/Retina1.png)

