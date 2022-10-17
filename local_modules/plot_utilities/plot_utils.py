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

