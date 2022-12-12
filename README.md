# Using Deep-Learning to classify individual ponds of small scale gold mining in Ghana

In this project a classification tool was developed that can identify and categorise gold mining ponds usng Sentinel-2 images. 

All of the nessesary data preprocessing is stored in the preprocesssor module within the Tools package. Function create_model_dataset encapsulates all the tasks needed to prepare training data, so it is the only one used in Workflow_Notebook. However, one can refer to the docstrings in each helper function for the detailed descriptions of tasks performed. If re-training models with other samples, one can use the data loading functions available in this project, given the folder structure is the following:

data/training_inputs/sample1/1.tiff
                     sample2/2.tiff
                     sample3/3.tiff
                     samplex/x.tiff
    /training_labels/1.png
                    /2.png
                    /3.png
                    /x.png
In the data for this project fold

Visualiser module in tools package  plots the inputs, targets and outputs of the model. Lastly ghana.py is a module that read-in Sentinel-2 images of Ghana and preprocess them with additional step of histogram matching (see Methodology/Spatial and Temporal Trend Analysis section for detailed explanation of how histogram matching works)

Unet package contains core module - Fri_unet.py used to initialise the model. The structure was adopted from open-source tutorial https://github.com/johschmidt42/PyTorch-2D-3D-UNet-Tutorial. The customdataset.py module is used to create torch datasets, trainer to initialise the training process and focal.py is the Focal Loss function. 

The Workflow_Notebook.ipynb can be used to see how functions and packages are used chronologically to reach the results presented in report. Otherwise, if workflows need to be apdapted for othe problems, all core functions have detailed docstrings explaining the inputs taken, tasks performed and outputs, hence can be re-used. 

The models and data can be downloaded from this DropBox folder:
https://www.dropbox.com/sh/6mq9sc41w46l5n4/AABxdlLX7NNI2ghBv-mPg0cra?dl=0
The models are sorted according to the number of output classes (Binary, 3 classes, 4 classes)
And the data is stored in data folder where Bigger_Set_21_19_HistMatch is input data and Thr_Labels_final is target label data. In sub-folder Ghana there are 4 Sentinel-2 images and another sub-folder Planet, for 3-channel basemaps from planet.com.

To download Ghana Sentinel-2 images and cloud mask them,  this code in Google Earth Engine can be used:
https://code.earthengine.google.com/677781786af4185c59e8f6556f745ad2
It could be-nessesary to re-import cloud mask dataset, which is provided in this repository - Cloud_Mask_Dataset.csv

!Important!
When re-loading models it is nessesary to put them inside Unet package.
