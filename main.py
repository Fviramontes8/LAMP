import matplotlib.pyplot as plt
import numpy as np

import Test_python

from local_modules.data_io.mat_io import open_mat_file
from local_modules.plot_utilities.plot_utils import create_subplots

def main() -> None:
    relay_data_dict = open_mat_file("Data/sample_C1RTL3.mat")
    data_key = "sample_C1RTL3"
    feature_labels = [
        "Timestamp",
        "V1",
        "V2",
        "V0",
        "I1",
        "I2",
        "I0",
        "Fault Status (0, 1)?",
        "Fault Type (1 - 4)?",
        "Configuration?",
        "Fault Resistance (0 - 1)?",
        "?",
    ]

    relay_data_array = np.array(relay_data_dict[data_key]).T
    create_subplots(relay_data_array, feature_labels, 3, 4)

    testing_data = relay_data_array[1:7]
    print(f"Testing data size: {testing_data.shape}")


if __name__ == "__main__":
    main()
