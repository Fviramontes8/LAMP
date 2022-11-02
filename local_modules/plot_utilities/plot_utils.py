import matplotlib.pyplot as plt
import numpy as np


"""
@brief Supporting function to help visualize the relay data

@param data -- Source data to plot
@param labels -- Labels to describe data, must be the same size as data
@param dim0 -- Number of rows for the subplots
@param dim1 -- Number of columns for the subplots
"""
def create_subplots(data: np.array, labels: list, 
        dim0: int, dim1: int) -> None:
    assert(len(data) == len(labels))
    x_data = [ i for i in range(len(data[0])) ]
    fig, ax = plt.subplots(dim0, dim1, sharex=True);
    fig.suptitle("Solar relay timeseries of 200 samples")

    for i in range(len(data)):
        ax_row = i // ax.shape[1]
        ax_col = i % ax.shape[1]
        ax[ax_row][ax_col].plot(x_data, data[i])
        ax[ax_row][ax_col].set_title(labels[i])
    plt.show()


"""
@brief Same functionality as create_subplots but rather than opening a 
    window to show the data, it will save the figure as a png

@param data -- Source data to plot
@param labels -- Labels to describe data, must be the same size as data
@param png_filename -- Filename to save the figure as a png
@param dim0 -- Number of rows for the subplots
@param dim1 -- Number of columns for the subplots
"""
def save_subplots(data: np.array, labels: list, png_filename: str,
        dim0: int, dim1: int) -> None:
    assert(len(data) == len(labels))

    x_data = [ 
        i for i in range(len(data[0])) 
    ]
    fig, ax = plt.subplots(dim0, dim1, sharex=True);
    fig.suptitle("Solar relay timeseries of 200 samples")

    for i in range(len(data)):
        ax_row = i // ax.shape[1]
        ax_col = i % ax.shape[1]
        ax[ax_row][ax_col].plot(x_data, data[i])
        ax[ax_row][ax_col].set_title(labels[i])
    plt.savefig(png_filename)


"""
@brief Takes a time series of configuration predictions and saves a plot to be
    seen in a separate image viewing program

@param cnn_history -- A time series containing a history of configuration
    predictions
@param title -- Title to be used in the graph
"""
def save_cnn_history(cnn_history: np.array, 
        title="Past 100 configurations given a data sample") -> None:
    data_pts = 100
    if cnn_history.shape[0] < 100:
        data_pts = cnn_history.shape[0]
        title = f"Past {data_pts} configurations given a data sample"

    x_data = [
        i+1 for i in range(data_pts)
    ]

    plt.plot(x_data, cnn_history[-data_pts:, 0], "b-", 
        label="Configuration 1 probability")
    plt.plot(x_data, cnn_history[-data_pts:, 1], "r-", 
        label="Configuration 2 probability")
    plt.plot(x_data, cnn_history[-data_pts:, 2], "g-", 
        label="Configuration 3 probability")
    plt.plot(x_data, cnn_history[-data_pts:, 3], "o-", 
        label="Configuration 4 probability")
    plt.title(title)
    plt.legend()
    plt.savefig("CNN_History.png")
    plt.clf()


"""
@brief Takes a time series of probability fault predictions and saves a plot 
    to be seen in a separate image viewing program

@param pf_history -- A time series containing a history of probability fault
    predictions
@param title -- Title to be used in the graph
"""
def save_pf_history(pf_history: list, 
        title="Past 100 fault probabilities given a data sample") -> None:
    data_pts = 100
    if len(pf_history) < 100:
        data_pts = len(pf_history)
        title = f"Past {data_pts} fault probabilities given a data sample"

    x_data = [
        i+1 for i in range(data_pts)
    ]

    plt.plot(x_data, pf_history[-data_pts:], "c-", 
        label="Fault probability")
    plt.title(title)
    plt.legend()
    plt.savefig("PF_History.png")
    plt.clf()


"""
@brief Takes a time series of probability fault configuration predictions and 
    saves a plot to be seen in a separate image viewing program

@param pf_history -- A time series containing a history of probability fault
    configuration predictions
@param title -- Title to be used in the graph
"""
def save_pfc_history(pfc_history: np.array, 
        title="Past 100 configurations given a data sample") -> None:
    data_pts = 100
    if pfc_history.shape[0] < 100:
        data_pts = pfc_history.shape[0]
        title = f"Past {data_pts} configurations given a data sample"

    x_data = [
        i+1 for i in range(data_pts)
    ]

    plt.plot(x_data, pfc_history[-data_pts:, 0], "k-", 
        label="Fault configuration 1 probability")
    plt.plot(x_data, pfc_history[-data_pts:, 1], "m-", 
        label="Fault configuration 2 probability")
    plt.plot(x_data, pfc_history[-data_pts:, 2], "y-", 
        label="Fault configuration 3 probability")
    plt.title(title)
    plt.legend()
    plt.savefig("PFC_History.png")
    plt.clf()


