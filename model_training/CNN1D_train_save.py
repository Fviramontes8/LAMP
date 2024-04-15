#!/usr/bin/env python
# coding: utf-8


import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, Dropout, Flatten
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
import time
import itertools


def load_train_Data(relay='RTL3'):
    os.chdir('../Data')
    data1 = sio.loadmat(f'{relay}.mat')['C1' + relay]
    data1[:, 1:7] = (data1[:, 1:7] - data1[:, 1:7].min(axis=0)) / (
        data1[:, 1:7].max(axis=0) - data1[:, 1:7].min(axis=0)
    )
    data2 = sio.loadmat(f'{relay}.mat')['C2' + relay]
    data2[:, 1:7] = (data2[:, 1:7] - data2[:, 1:7].min(axis=0)) / (
        data2[:, 1:7].max(axis=0) - data2[:, 1:7].min(axis=0)
    )
    data3 = sio.loadmat(f'{relay}.mat')['C3' + relay]
    data3[:, 1:7] = (data3[:, 1:7] - data3[:, 1:7].min(axis=0)) / (
        data3[:, 1:7].max(axis=0) - data3[:, 1:7].min(axis=0)
    )
    data4 = sio.loadmat(f'{relay}.mat')['C4' + relay]
    data4[:, 1:7] = (data4[:, 1:7] - data4[:, 1:7].min(axis=0)) / (
        data4[:, 1:7].max(axis=0) - data4[:, 1:7].min(axis=0)
    )

    return data1, data2, data3, data4


def convert_to_3d(data, n_time=100):
    a1 = int(np.floor(data.shape[0] / n_time) * n_time)
    data = data[:a1, :]
    n_feat = data.shape[1]
    X1 = data.reshape((round(data.shape[0] / n_time), n_time, n_feat))
    X11 = X1[:, :, 1:7]
    Y11 = X1[:, 0, 11]
    return X11, Y11


def data_preparation(data1, data2, data3, data4):
    [data11, lab1] = convert_to_3d(
        data1
    )   # organizing into a 3d data with time frames
    [data22, lab2] = convert_to_3d(data2)
    [data33, lab3] = convert_to_3d(data3)
    [data44, lab4] = convert_to_3d(data4)
    # print(data11.shape, lab1.shape, data22.shape, lab2.shape, data33.shape,
    #     lab3.shape, data44.shape, lab4.shape)

    train1 = data11[0: round(data11.shape[0] / 2), :, :]
    lab_train1 = lab1[0: round(data11.shape[0] / 2)]
    test1 = data11[round(data11.shape[0] / 2):, :, :]
    lab_test1 = lab1[round(data11.shape[0] / 2):]

    train2 = data22[0: round(data22.shape[0] / 2), :, :]
    lab_train2 = lab2[0: round(data22.shape[0] / 2)]
    test2 = data22[round(data22.shape[0] / 2):, :, :]
    lab_test2 = lab2[round(data22.shape[0] / 2):]

    train3 = data33[0: round(data33.shape[0] / 2), :, :]
    lab_train3 = lab3[0: round(data33.shape[0] / 2)]
    test3 = data33[round(data33.shape[0] / 2):, :, :]
    lab_test3 = lab3[round(data33.shape[0] / 2):]

    train4 = data44[0: round(data44.shape[0] / 2), :, :]
    lab_train4 = lab4[0: round(data44.shape[0] / 2)]
    test4 = data44[round(data44.shape[0] / 2):, :, :]
    lab_test4 = lab4[round(data44.shape[0] / 2):]

    # print(train1.shape, lab_train1.shape, train2.shape, lab_train2.shape,
    #       train3.shape, lab_train3.shape, train4.shape, lab_train4.shape)
    # print(test1.shape, lab_test1.shape, test2.shape, lab_test2.shape,
    #       test3.shape, lab_test3.shape, test4.shape, lab_test4.shape)
    X_train = np.concatenate((train1, train2, train3, train4), axis=0)
    Y_train = np.concatenate(
        (lab_train1, lab_train2, lab_train3, lab_train4), axis=0
    )
    x_test_orig = np.concatenate((test1, test2, test3, test4), axis=0)
    y_test_orig = np.concatenate(
        (lab_test1, lab_test2, lab_test3, lab_test4), axis=0
    )

    return X_train, Y_train, x_test_orig, y_test_orig


def convert_tocategorical(y, number_of_classes=4):
    y_conv = np_utils.to_categorical(y.ravel() - 1, number_of_classes)
    return y_conv


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 1, 32
    n_timesteps, n_features, n_outputs = (
        trainX.shape[1],
        trainX.shape[2],
        trainy.shape[1],
    )
    model = Sequential()
    model.add(
        Conv1D(
            filters=64,
            kernel_size=3,
            activation='relu',
            input_shape=(n_timesteps, n_features),
        )
    )
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
    )
    # fit network
    model1 = model.fit(
        trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose
    )
    # evaluate model
    _, accuracy = model.evaluate(
        testX, testy, batch_size=batch_size, verbose=verbose
    )
    return accuracy, model, model1


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_experiment(X_train, Y_train, X_test, Y_test, repeats=10):
    # load data
    trainX = X_train
    trainy = Y_train
    testX = X_test
    testy = Y_test
    max_acc = 0
    # Split the dataset in two equal parts
    # X_train, X_val, Y_train, y_val = train_test_split(
    #     X_train, Y_train, test_size=0.2, random_state=0)
    # repeat experiment
    scores = list()
    for r in range(repeats):
        [score, model, model1] = evaluate_model(
            trainX, trainy, testX, testy
        )  # , X_val, y_val)
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
        if score > max_acc:
            best_model = model
            max_acc = score
        # summarize results
    summarize_results(scores)
    return scores, best_model, max_acc, model1, model


def plot_confusion_matrix(
    cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Purples
):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm * 100, decimals=2)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        print('Normalized confusion matrix')
    else:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment='center',
            color='white' if cm[i, j] > thresh else 'black',
        )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return (TP, FP, TN, FN)


def main():
    relay = 'RTL3'
    [data1, data2, data3, data4] = load_train_Data(relay)
    [X_train, ytrain, X_test, ytest] = data_preparation(
        data1, data2, data3, data4)
    Y_train = convert_tocategorical(ytrain, 4)
    Y_test = convert_tocategorical(ytest, 4)
    start = time.time()
    [scores, best_model, max_acc, model1, model] = run_experiment(
        X_train, Y_train, X_test, Y_test
    )
    end = time.time()
    print(end - start)
    Y_pred = best_model.predict(X_test, verbose=2)
    # Y_pred.shape
    y_pred = np.argmax(Y_pred, axis=1)

    for ix in range(4):
        print(ix, confusion_matrix(
            np.argmax(Y_test, axis=1), y_pred)[ix].sum())
    cm = confusion_matrix(np.argmax(Y_test, axis=1), y_pred)
    print(cm)
    scores = np.array(scores) / 100
    cnn = model1

    f2 = plt.figure(0)
    plt.plot(cnn.history['accuracy'], 'r')
    plt.plot(scores, 'g')
    plt.xticks(np.arange(0, 11, 2.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel('Num of Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.legend(['train', 'validation'])
    f2.savefig('accuracy.png', bbox_inches='tight')
    plt.show()

    [TP, FP, TN, FN] = perf_measure(np.argmax(Y_test, axis=1), y_pred)
    print(TP, FP, TN, FN)

    p_detect = TP / (TP + FN)
    p_false = FP / (TN + FN)
    print(p_detect, p_false)

    # PLOT CONFUSION MATRIX (TEST)
    target_names = ['Config1', 'Config2', 'Config3', 'Config4']
    np.set_printoptions(precision=2)
    cnf_matrix = cm
    print(max_acc)
    print(p_detect, p_false)
    class_names = target_names
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(
        cnf_matrix,
        classes=class_names,
        title='Confusion matrix without normalization',
    )

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(
        cnf_matrix,
        classes=class_names,
        normalize=True,
        title='Normalized confusion matrix',
    )

    plt.show()

    os.chdir('../Models')
    best_model.save_weights('CNN_weights_RTL3.h5')


if __name__ == "__main__":
    main()
