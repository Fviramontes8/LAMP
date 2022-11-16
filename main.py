import matplotlib.pyplot as plt
import numpy as np
import time

import local_modules.plot_utilities.plot_utils as pu
from local_modules.data_io.mat_io import open_mat_file
from local_modules.model_inference import fault_detector

def main() -> None:
    relay_data_dict = open_mat_file("Data/RTL1_ALL_OPAL.mat")
    data_key = "all_quantities"
    relay_name = "RTL3"
    feature_labels = [
        "Timestamp",
        "V1",
        "V2",
        "V0",
        "I1",
        "I2",
        "I0",
        "Configuration",
        "Fault Status",
        "Fault Type (1 - 3)",
        "Fault Location (Fault bus 20)",
    ]

    relay_data_array = np.array(relay_data_dict[data_key]).T
    relay_data_length = len(relay_data_array[0])
    slice_len = 500000
#    pu.create_subplots(
#        relay_data_array[:10, :slice_len], 
#        feature_labels[:10], 
#        2, 
#        5
#    )

    p_cnn_history = np.zeros((1, 4))
    pf_history = []
    pfc_history = np.zeros((1, 3))

    # CNN component requires 101 values before it can begin inferencing
    #   on new data
    total_testing_data = [ relay_data_array[1:7, i] for i in range(101) ]
    for i in range(10000, slice_len): #relay_data_length):
        testing_data_slice = list(relay_data_array[1:7, i])
        total_testing_data.append(testing_data_slice)
        p_cnn, pf, pfc = fault_detector.compute_fault_probabilities(
            testing_data_slice,
            total_testing_data,
            relay_name
        )
        print(f"Iteration {i-101}, p_cnn: {p_cnn}, pf: {pf}, pfc: {pfc}")

        p_cnn_history = np.vstack((p_cnn_history, p_cnn))
        pf_history.append(pf)
        pfc_history = np.vstack((pfc_history, pfc))
        pu.save_cnn_history(p_cnn_history)
        pu.save_pf_history(pf_history)
        pu.save_pfc_history(pfc_history)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
