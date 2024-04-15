#!/usr/bin/env python
# coding: utf-8

import sys
import warnings
import torch
import gpytorch
from gpytorch.priors import GammaPrior
from matplotlib import pyplot as plt
import scipy.io as sio
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import svm
import pickle
import os


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(
        #     variance_prior=gpytorch.priors.GammaPrior(0.5, 2)))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel(variance_prior=GammaPrior(2, 2))
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def GP_model(train_x, train_y):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(1.0),
        'covar_module.outputscale': torch.tensor(2.0),
    }

    model = ExactGPModel(train_x, train_y, likelihood)
    model.initialize(**hypers)
    # print(
    #  model.likelihood.noise_covar.noise.item(),
    #   model.covar_module.outputscale.item()
    # )

    # print(model.named_parameters)
    # for param_name, param in model.named_parameters():
    # print(f'Parameter name: {param_name:42} value = {param.item()}')

    return model, likelihood


def GP_train(train_x, train_y, n_iter=2000):
    [model, likelihood] = GP_model(train_x, train_y)

    smoke_test = 'CI' in os.environ
    training_iter = 2 if smoke_test else n_iter
    model = model.double()

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    train_loss = np.zeros((training_iter, 1))
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        train_loss[i] = loss.item()
        # print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
        # i + 1, training_iter, loss.item(),
        # model.likelihood.noise.item()
        # ))
        optimizer.step()

    return model, likelihood, train_loss


def Gp_test(test_x, model, likelihood):
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # observed_pred = model(test_x)
        observed_pred = likelihood(model(test_x))
        f_var = observed_pred.variance

    return observed_pred, f_var


def plot_test_GP(observed_pred, f_var, test_y):

    n_sample0 = 0
    n_sample1 = test_y.shape[0]

    with torch.no_grad():
        # Initialize plot
        # Get upper and lower confidence bounds
        # lower, upper = observed_pred.confidence_region()

        # mae = mean_absolute_error(test_y.numpy(), observed_pred.mean.numpy())
        # mse = mean_squared_error(test_y.numpy(), observed_pred.mean.numpy())
        # mse_var = mean_squared_error(
        #     test_y.numpy(), observed_pred.mean.numpy()
        # ) / np.var(test_y.numpy(), axis=0)
        # lower = observed_pred.mean - 2 * f_var
        # upper = observed_pred.mean + 2 * f_var
        error1 = (
            observed_pred.mean.numpy()[n_sample0:n_sample1]
            - test_y.numpy()[n_sample0:n_sample1]
        )
    return error1, f_var


def probability_SC(true, pred):
    """
    returns the probability of detection
    and probability of false alarm
    """
    cm = confusion_matrix(true, pred)
    pdetect = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    pfalse = cm[1, 0] / (cm[1, 0] + cm[1, 1])

    return cm, pdetect, pfalse


def pdtect_vs_pfalse_swipe(scores_oc, y_true, step):
    threshold = []
    p_detect = []
    p_false = []
    cmat = []
    for thresh in np.arange(min(scores_oc), max(scores_oc) + 0.01, step):
        y_pred = 2 * (scores_oc > thresh) - 1
        [cm, pd, pf] = probability_SC(y_true, y_pred)
        threshold.append(thresh)
        p_detect.append(pd)
        p_false.append(pf)
        cmat.append(cm)
    return p_detect, p_false, threshold, cmat


def OCSVM_train(train):
    clf = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)
    clf.fit(train)

    return clf


def OCSVM_test(clf, test, test_lab):
    pred = clf.predict(test)
    scores_oc = clf.decision_function(test)
    p = 1 / (1 + np.exp(scores_oc))

    return pred, scores_oc, p


def swipe_plot(scores_oc, test_lab, step):
    [pdetect_s, pfalse_s, thresh_s, cmat] = pdtect_vs_pfalse_swipe(
        scores_oc, test_lab, step
    )
    plt.plot(pfalse_s, 1 - np.array(pdetect_s), '.')
    plt.xlabel('PFA')
    plt.ylabel('1-PD')
    plt.grid()


def GP_plus_OC_TRAIN(train_x, train_y, train_iter, tr_switch):
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)

    # TRAIN
    # GP1 train
    [model1, likelihood1, loss1] = GP_train(
        train_x.double(), train_y[:, 0].double(), train_iter
    )
    [observed_pred11, f_var11] = Gp_test(train_x.double(), model1, likelihood1)
    [error11, f_var11] = plot_test_GP(
        observed_pred11, f_var11, train_y[:, 0].double()
    )

    # GP2 train
    [model2, likelihood2, loss2] = GP_train(
        train_x.double(), train_y[:, 1].double(), train_iter
    )
    [observed_pred22, f_var22] = Gp_test(train_x.double(), model2, likelihood2)
    [error22, f_var22] = plot_test_GP(
        observed_pred22, f_var22, train_y[:, 1].double()
    )

    # GP3 Train
    [model3, likelihood3, loss3] = GP_train(
        train_x.double(), train_y[:, 2].double(), train_iter
    )
    [observed_pred33, f_var33] = Gp_test(train_x.double(), model3, likelihood3)
    [error33, f_var33] = plot_test_GP(
        observed_pred33, f_var33, train_y[:, 2].double()
    )

    std_nf1 = np.sqrt(f_var11)
    std_nf2 = np.sqrt(f_var22)
    std_nf3 = np.sqrt(f_var33)

    train = np.vstack(
        (error11 / std_nf1, error22 / std_nf2, error33 / std_nf3)
    ).T

    clf = OCSVM_train(train)

    return model1, model2, model3, clf


def GP_plus_OC_TEST(test_x, test_y, test_lab):
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)
    # TEST
    # test GP1
    [observed_pred1, f_var1] = Gp_test(test_x.double(), model1, likelihood1)
    [error1, f_var1] = plot_test_GP(
        observed_pred1, f_var1, test_y[:, 0].double()
    )

    # test GP2
    [observed_pred2, f_var2] = Gp_test(test_x.double(), model2, likelihood2)
    [error2, f_var2] = plot_test_GP(
        observed_pred2, f_var2, test_y[:, 1].double()
    )

    # test GP3
    [observed_pred3, f_var3] = Gp_test(test_x.double(), model3, likelihood3)
    [error3, f_var3] = plot_test_GP(
        observed_pred3, f_var3, test_y[:, 2].double()
    )

    std_f1 = np.sqrt(f_var1)
    std_f2 = np.sqrt(f_var2)
    std_f3 = np.sqrt(f_var3)

    test = np.vstack((error1 / std_f1, error2 / std_f2, error3 / std_f3)).T

    # print(train.shape, test.shape)

    [pred, scores_oc, p] = OCSVM_test(clf, test, test_lab)

    return pred, scores_oc, p, test


def loading_train_data(data1):
    """
    Data loading code for loading the data for GP1, GP2, GP3
    and testing data for the same
    """
    j = 0
    i = 2000
    s = 20

    train = data1[0: round(data1.shape[0] / 2), :]
    train_nf = train[np.where(train[:, 10] == 0)]
    i1 = round(train_nf.shape[0] / 2)
    i0 = i1 - 100000
    max1 = train_nf[i0:i1:s, 1:7].max(axis=0)
    min1 = train_nf[i0:i1:s, 1:7].min(axis=0)
    train_nf[i0:i1:s, 1:7] = (train_nf[i0:i1:s, 1:7] - min1) / (max1 - min1)
    train_x = train_nf[i0:i1:s, 1:4]
    train_y = train_nf[i0:i1:s, 4:7]   # for all GP
    train_lab = train_nf[i0:i1:s, 10]

    return train_x, train_y, train_lab, [max1, min1]


def main():
    if not sys.warnoptions:
        warnings.simplefilter('ignore')
    warnings.filterwarnings('ignore')

    dir_path = 'RTL3.mat'
    mat_file = 'C4RTL3'
    train_iter = 2000
    tr_switch = 1
    os.chdir('../Data')
    data1 = sio.loadmat(dir_path)[mat_file]
    train_x, train_y, train_lab, max_min_tr = loading_train_data(data1)

    [GP1, GP2, GP3, OCSVM] = GP_plus_OC_TRAIN(
        train_x, train_y, train_iter, tr_switch
    )

    # GP1.state_dict()
    # GP2.state_dict()
    # GP3.state_dict()

    os.chdir('../Models')

    torch.save(GP1.state_dict(), 'GP1_' + mat_file + '.pth')
    torch.save(GP2.state_dict(), 'GP2_' + mat_file + '.pth')
    torch.save(GP3.state_dict(), 'GP3_' + mat_file + '.pth')

    file_name = 'OCSVM_' + mat_file + '.sav'
    pickle.dump(OCSVM, open(file_name, 'wb'))
    np.save('maxmin_GPOCSVM_' + mat_file + '.npy', max_min_tr)
    np.save('train_GP_' + mat_file + '.npy', [train_x, train_y])


if __name__ == "__main__":
    main()
