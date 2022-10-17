"""
@author Aswathy R Kurup

Original code was written by Dr. Aswathy R Kurup and modified by 
    Francisco Viramontes to be used as a module
"""

import pickle
import os
from sklearn.svm import SVC
import scipy.io as sio
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def data_preparation(data2, max1, min1):
    return (data2 - min1) / (max1 - min1)


'''
    LOADING MODEL PARAMETERS:
    - loading the svm model 
    - loading max and min values of the training data
    for normalization.
'''
def load_svm_model(config, relay):
    # config = 'C1'
    fil_prefix = f"{os.getcwd()}/Models/"
    fil = f"{config}{relay}"
    filename = f"{fil_prefix}{fil}_svmmodel.sav"
    loaded_model = pickle.load(open(filename, 'rb'))

    max_min = np.load(f"{fil_prefix}maxmin_svm_{fil}.npy")

    return loaded_model, max_min


'''
    Testing the data using the loaded svm model
'''
def svm_test(x_test_orig, svm_model):
    svm_predictions = svm_model.predict(x_test_orig)
    scores_oc = svm_model.decision_function(x_test_orig)
    prob = np.exp(scores_oc)/np.sum(np.exp(scores_oc),axis=1, keepdims=True) # softmax after the voting
    return prob


def loading_matfile_test(dir_path, relay, config):
    dir_sample = '../Data'
    os.chdir(dir_sample)
    mat_file = config+relay 
    data2 = sio.loadmat(relay+'.mat')[mat_file]
    test = data2[round(data2.shape[0]/2):,:]
    return data2


def loading_Sample_file(relay, config):
    mat_file = 'sample_' + config + relay
    dir_sample = '../Data'
    os.chdir(dir_sample)
    data2 = sio.loadmat(mat_file)[mat_file]
    return data2


"""
@brief Main driver function to load weights into an svm and using the loaded
    weights to run inference on a provided dataset.

@param data - Source data to run interence on
@param config - Solar array configuration to run inference on
@param relay - Solar relay configuration that data belongs to
"""
def svm_test_main(data, config, relay):
    [svm_model, max_min_tr] = load_svm_model(config, relay)
# using pymodbus
    test = data_preparation(data,  max_min_tr[0,:],  max_min_tr[1,:])
#### CHANGED HERE ####    
    prob = svm_test(np.array(test).reshape(-1,1).T, svm_model)
    return prob[0] # probability has the probability scores
