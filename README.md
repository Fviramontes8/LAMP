# LAMP
This git is forked from aswathyrk93/LAMP_CODE.

There are 4 folders in this repository:

- Data
- Models
- Test_python
- Train

## Installing dependencies
The pip dependencies can be installed with requirements.txt. The following command can be used to install the dependencies:

```
pip install -r requirements.txt
```


## Data
Contains the sample data for each configuration and also the train data that will be used for re-initializing the Gaussian Process (GP) model.

## Models

Contains all the models that were trained offline. These include:

- Gaussian Processes (GP)
- Support Vector Machines (SVM)
- One Class SVMs (OCSVM)
- Convolutional Neural Networks (CNN).

The names of the models describe the configuration that they are trained on, as well as the name of the model. For example, `OCSVM_C2RTL3.sav` describes a One Class Support Vector Machine (OCSVM) that was trained on the RTL3 relay when it tuned to configuration 2 (C2).

These models include the maximum and the minimum values that is used for normalization in case of GP + OCSVM and SVM classifiers.

These models are loaded in test set up

## model_inference
The folder contains 4 files of importance. The file `fault_detector.py` drives the other files in the folder:

   - cnn_fault_detector.py : tests the CNN block for classifying the configurations.
   - svm_fault_classifier.py : Tests the SVM model trained using data of four configurations. The configuration can be specified as a string (C1, C2, C3, C4) as the second parameter when calling the function `svm_test_main`.
   - gp_ocsvm_fault_detector.py: Tests the GP_OCSVM model trained using data of various configurations. Like with `svm_fault_classifier.py`, the various can be specified as a string (C1, C2, C3, C4) as the second parameter in the function `GP_test_main`.
   - fault_detector.py: Contains various experiments that read data and prepare to pass the data into the combination of models to perform the final computation of probabilities. 

The main code here is the `main.py` which reads the data from client and processes them through all the models in `MODELS` folder using `fault_detector.py` and computes the final output to save them as graphs.

## Train
Contains ipynb files for training all the algorithms: CNN (for topology estimation), GP+OCSVM (fault detection), SVM (fault classification)

The train data needs to be accessed from outside. These files were trained using a huge file called 'RTL3.mat" which had data corresponding to all the configuration. These data is used for training all the algorithms.

The folder contains 6 files:

   - `CNN1D_train_Save`: trains the CNN model and saves the model in ".h5" format.
   - `Fault_classifier_trainC1`:  trains the fault classifier using SVM for configuration 1 and saves the corresponding model. Similarly we have fault classifier training models for the other three configurations
   - `GP_OCSVM_train`: **THIS FILE IS IMPORTANT.** This mode needs to RETRAINED most frequently. EVERY other day. This is unsupervised block and it trains on the input data and saves all three GP models and OCSVM model



            
