# Retinal Vessel Segmentation
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://linkedin.com/in/sandeep-ghosh)


A subpixel dense and deeply supervised U-Net like architecture for segmenting retinal vessels.

## Overview
* [Contributors](#Contributors)
* [Introduction](#Introduction)
* [Dataset Used](#Dataset-Used)
* [Augmentation & Preprocessing](#Augmentation-and-Preprocessing)
* [Network Architecture](#Network-Architecture)
* [Loss Function & Optimizer](#Loss-Function-and-Optimizer)
* [Training setup](#Training-setup)
* [Evaluation Metric](#Evaluation-Metric)
* [Results](#Results)

### Contributors:
This project is created by the joint efforts of
* [Subham Singh](https://github.com/Subham2901)
* [Sandeep Ghosh](https://github.com/Sandeep2017)

### Introduction:
Automatic extraction of retinal vessels from retinal fundas images is a long researched area in AI aided medical diagnosis as it assists in the implementation of screening programs for diabetic retinopathy, to establish a relationship between vessel tortuosity and hypertensive retinopathy, vesssel diameter measurement in relation with diagnosis of hypertension, and computer assisted laser surgery. Also, the retinal vascular tree is found to be unique for each individual and can be used for biometric identification. 
For all the above mentioned reasons and for many others, automated retinal vessel extraction is a very important task. In this project, we have tried to use a modified [U-Net](https://arxiv.org/abs/1505.04597) like structure and achieved near state-of-the-art results. 

### Dataset Used:
The [DRIVE dataset](https://drive.grand-challenge.org/) is used here for training as well as testing purposes.
The dataset consists of 40 expert annotated retinal fundas images divided equally into a training and test set of 20 image each. Each image was captured using 8 bits per color plane at 768 by 584 pixels. The FOV of each image is circular with a diameter of approximately 540 pixels. For this dataset, the images have been cropped around the FOV. For each image, a mask image is provided that delineates the FOV. 
A sample fundas image along with its hand annotated mask is provided below.

Original Image             |  Mask Image               |        AV Mask
:-------------------------:|:-------------------------:|:-------------------------:|
![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/29-training.png)  |  ![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/29_training.png)   |   ![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/29_training%20(2).png)

### Augmentation and Preprocessing:

The training data was augmented on the fly using the [Albumentations library](https://albumentations.ai/).
A strong combination of different types of image augmentations were applied with varied probabilities. They were:
* Random Flips.
* Transpose.
* Scale, Shift & rotate.
* Random Rotations.
* Optical Distortion.
* Grid Distortion.
* Elastic Transform.
* RGB Shift.
* Random Gamma.
* Random Brightness & contrast.

Along with the above mentioned augmentations, every image in the training and testing sets underwent a Histogram Equalization preprocessing step, i.e, CLAHE (Contrast Limited Adaptive Histogram Equalization).

Some examples of augmented images and masks are given below.

Augmented Training Images
:-------------------------:|
![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/x.PNG)

Augmented Training Masks
:-------------------------:|
![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/y.PNG)|

[Back to top](#Retinal-Vessel-Segmentation)

### Network Architecture:
* The proposed architecture shown in Fig. 1, below is inspired from the original U-Net architecure with major modifications. It is a 4 stage segmentation architecture based upon the concept of stacking. In this 4 stage architecture, the feature map from the penultimate layer of the first stage is passed on to the second stage as the penultimate layer contains more information than the last output layer. The passed on feature map is then concatenated with the original input shape (shown in bold black arrows) and is fed as an input to the second stage. Concatenating the original shape image with the previous feature map improves the refinement of the predictions. Similarly, the feature map from the penultimate layer of the second stage is passed on the third stage and concatenated with the original input image as so on until the fourth stage. The final predictions are obtained from the output of the fourth stage.

* Here the learnable transposed convolution method or the interpolation based up-sampling method is replaced by the efficient [subpixel convolutions](xxxxx).
A subpixel convolutional layer is shown in Fig. 2, which is just a standard 1x1 convolution layer followed by a pixel shuffling operation which re-arranges the pixels from the depth dimensions to the spatial dimensions. This type pf convolutions minimize information loss during up-sampling of the image in the decoder network.

* Each stage has its own output layer and the loss is minimized at each stage. The architecture is trained using deep supervision.

* The parallel layers of the four decoder networks are connected via dense connections which helps in improving the spatial knowledge transfer through each stage of the model while training.

Fig. 1 Network architecture | Fig. 2 Subpixel Convolutional layer
:-------------------------:|:-------------------------:|
![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/Retina1.png) | ![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/subpixel2.png)

Fig. 3 Encoder Block
:-------------------------:|
![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/Encoder1.png)

[Back to top](#Retinal-Vessel-Segmentation)

### Loss Function and Optimizer:

#### Loss Function
Pixel-wise Binary Cross-entropy is a widely used loss function for semantic segmentation since it evaluates the class predictions for each pixel individually, but it suffers from class imbalance, so we have used another loss function along with BCE loss, named Dice loss which has a normalizing effect and is not affected by class imbalance. 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;BCELoss={-[ylogp+(1-y)log(1-p)]}"  />


Dice Loss = 2 ×| A ∩ B |/| A ∪ B |

#### Optimizer
Adam optimizer was used with default parameters

### Training setup:
* GPU: Nvidia P100 16GB
* CPU: Intel Xeon
* RAM: 12GB DDR4

The network was trained using the above mentioned setup for 40 epochs with a batch size of ```10``` and input image size ```256 x 256 x 3```. Total time taken for training is 1.5 hours

[Back to top](#Retinal-Vessel-Segmentation)

### Evaluation Metric:
F1-score was used for evaluation as it takes into account the class imbalance, because the number of vessel pixels greatly outnumbers the non-vessel pixels.
It is the harmonic mean of precision and recall.

<img src="https://latex.codecogs.com/svg.latex?\Large&space;F1-Score={2*(Precision*Recall)/(Precision+Recall)}"  />

The table below compares the F1-score, AUC and the IoU of all the stages.

Stage|F1-Score | AUC | IoU | Dice coef.|
---|--- | --- | --- | --- |
Stage 1|84.0 | 90.0 | 74.0 | -- | 
Stage 2|85.5 | 91.7 | 74.2 | -- | 
Stage 3|86.3 | 92.0 | 75.6 | -- | 
**Stage 4**|**87.5** | **94.7** | **77.2** | **--** |
--|-- | -- | -- | -- |
Average|87.8 | 92.1 | 75.2 | -- |

The table below compares our method with existing methods.

Model|F1-Score | AUC | IoU | Dice coef.|
---|--- | --- | --- | --- |
U-Net| | | |
U-Net++| | | |
Wide U-Net| | | |
Residual U-Net| | | |
Pretrained VGG encoded U-Net| | | |
Pretrained MobileNetV2 encoded U-Net| | | |
Pretrained ResNet encoded U-Net| | | |
**Proposed Method**|**87.5**|**94.7**|**77.2**|**--**|

[Back to top](#Retinal-Vessel-Segmentation)




### Results:
The predicted masks from each stage are shown below along with their respective ground truth masks and the original fundus images. 
Notice that the predictions from stage 4 are the best.
Original Fundus images
:-------------------------:|
![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/Results/original.PNG)

Original Ground truth masks
:-------------------------:|
![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/Results/original%20mask.PNG)

Stage 1 Predictions
:-------------------------:|
![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/Results/only%201.PNG)

Stage 2 predictions
:-------------------------:|
![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/Results/on%3By%202.PNG)

Stage 3 predictions
:-------------------------:|
![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/Results/only%203.PNG)

Stage 4 predictions
:-------------------------:|
![](https://github.com/Sandeep2017/Retinal-Vessel-Segmentation/blob/master/img/Results/Average.PNG)

[Back to top](#Retinal-Vessel-Segmentation)




