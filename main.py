import matplotlib.pyplot as plt
import numpy as np

from local_modules.data_io.mat_io import open_mat_file
from local_modules.plot_utilities.plot_utils import create_subplots
from local_modules.model_inference import fault_detector

from model_inference.microgrid_reading import compute_fault_probabilities


def main() -> None:
    relay_data_dict = open_mat_file("Data/sample_C1RTL3.mat")
    data_key = "sample_C1RTL3"
    relay_name = "RTL3"
    feature_labels = [
        "Timestamp",
        "V1",
        "V2",
        "V0",
        "I1",
        "I2",
        "I0",
        "Breaker Status",
        "Fault Impedance",
        "Fault Location (Fault bus 20)",
        "Fault Type (1 - 4)",
        "Configuration",
    ]

    relay_data_array = np.array(relay_data_dict[data_key]).T
    relay_data_length = len(relay_data_array[0])
    # create_subplots(relay_data_array, feature_labels, 3, 4)

    testing_data_slice = []
    total_testing_data = []
    for i in range(relay_data_length):
        testing_data_slice = np.atleast_2d(relay_data_array[1:7, i])
        total_testing_data.append(testing_data_slice)
        print(f"Data slice shape: {testing_data_slice.shape}")
        print(f"total testing data shape: {len(total_testing_data)}")
        p_cnn, pf, pfc = fault_detector.compute_fault_probabilities(
            testing_data_slice,
            total_testing_data,
            relay_name
        )
        print(f"p_cnn: {p_cnn}, pf: {pf}, pfc: {pfc}")


if __name__ == "__main__":
    main()
