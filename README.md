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

#### Image Augmentation

Augmented Training Images
:-------------------------:
![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/x.PNG)|

Augmented Training Masks
:-------------------------:
![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/y.PNG)|




### Network Architecture
![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/Retina1.png)

