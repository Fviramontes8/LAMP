from pymodbus.client.sync import ModbusTcpClient
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadDecoder
import time
from . import svm_fault_classifier as svm_fc
from . import gp_ocsvm_fault_detector as gp_svm_fd
from . import cnn_fault_detector as cnn_fd
from ..plot_utilities import plot_utils as pu
from ..data_io import mat_io
import numpy as np
import sys


"""
@brief Given a set of six features (V1, V2, V0, I1, I2, I0), they will be
    processed by 3 components: 4 Support Vector Machines (SVM), 4 Gaussian 
    Processes (GP) into 4 additional SVMs, and a Convolutional Neural Network 
    (CNN). Each component will generate probabilities and will be used to 
    calculate marginal probabilities for fault detection and fault 
    classification.

@param l1 -- Array like list containing six features
@param l11 -- Array like list containing the history of previous l1s
@param relay -- The relay configuration denoted as a string

TODO: Add documentation on return parameters
@return
"""
def compute_fault_probabilities(l1: list, l11: list, relay: str):
        # ->Tuple[float, float, float]:
    pfc_C1 = svm_fc.svm_test_main(l1, "C1", relay)
    pfc_C2 = svm_fc.svm_test_main(l1, "C2", relay)
    pfc_C3 = svm_fc.svm_test_main(l1, "C3", relay)
    pfc_C4 = svm_fc.svm_test_main(l1, "C4", relay)

    pGP_C1 = gp_svm_fd.GP_test_main(l1, "C1", relay)
    pGP_C2 = gp_svm_fd.GP_test_main(l1, "C2", relay)
    pGP_C3 = gp_svm_fd.GP_test_main(l1, "C3", relay)
    pGP_C4 = gp_svm_fd.GP_test_main(l1, "C4", relay)

    pCNN = cnn_fd.CNN_TEST1(np.array(l11), "C1", relay)

    # Computing the marginal probabilities for fault detector and fault classifier
    PFC1 = pfc_C1[0]*pCNN[0] + pfc_C2[0]*pCNN[1]\
        + pfc_C3[0]*pCNN[2] + pfc_C4[0]*pCNN[3]
    PFC2 = pfc_C1[1]*pCNN[0] + pfc_C2[1]*pCNN[1]\
        + pfc_C3[1]*pCNN[2] + pfc_C4[1]*pCNN[3]
    PFC3 = pfc_C1[2]*pCNN[0] + pfc_C2[2]*pCNN[1]\
        + pfc_C3[2]*pCNN[2] + pfc_C4[2]*pCNN[3]
    PFC = [PFC1, PFC2, PFC3]
    PF = pGP_C1*pCNN[0] + pGP_C2*pCNN[1] + pGP_C3*pCNN[2] + pGP_C4*pCNN[3]

    return pCNN, PF, PFC

def sample_c1rtl3_experiment():
    relay_data_dict = mat_io.open_mat_file("Data/sample_C1RTL3.mat")
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
    # pu.create_subplots(relay_data_array, feature_labels, 3, 4)

    p_cnn_history = np.zeros((1, 4))
    pf_history = []
    pfc_history = np.zeros((1, 3))

    # CNN component requires 101 values before it can begin inferencing
    #   on new data
    total_testing_data = [ relay_data_array[1:7, i] for i in range(101) ]
    for i in range(101, relay_data_length):
        testing_data_slice = list(relay_data_array[1:7, i])
        total_testing_data.append(testing_data_slice)
        p_cnn, pf, pfc = compute_fault_probabilities(
            testing_data_slice,
            total_testing_data,
            relay_name
        )
        #print(f"Iteration {i-101}, p_cnn: {p_cnn}, pf: {pf}, pfc: {pfc}")

        p_cnn_history = np.vstack((p_cnn_history, p_cnn))
        pf_history.append(pf)
        pfc_history = np.vstack((pfc_history, pfc))
        pu.save_cnn_history(p_cnn_history)
        pu.save_pf_history(pf_history)
        pu.save_pfc_history(pfc_history)
        time.sleep(0.5)

# [V1, V2, V0, I1, I2, I0, fault status_boolean (0 or 1), 
# fault type (2 fro AB as Adams doc), fault_resistance (0.5), 
# RTL1, RTL2, RTL3, RTL4]

def begin_modbus(ip_addr = "192.168.1.19", port=502):
    client = ModbusTcpClient(ip_addr, port)
    print(client.connect())
    print(client.is_socket_open())
    l11 = []
    i=0
    while client.connect():
        time.sleep(1)
        try:
            result = client.read_holding_registers(0, 26, unit=1) 
            decoder = BinaryPayloadDecoder.fromRegisters(
                result.registers,
                byteorder=Endian.Little,
                wordorder=Endian.Little
            )

            decoded_feature_names = [
                "Timestamp",
                "V1",
                "V2",
                "V0",
                "I1",
                "I2",
                "I0",
                "Configuration",
                "Fault Status",
                "Fault Type (1 - 4)",
                "Fault Location (Fault bus 20)",
            ]

            decoded_features = [
                decoder.decode_32bit_float(),
                decoder.decode_32bit_float(),
                decoder.decode_32bit_float(),
                decoder.decode_32bit_float(),
                decoder.decode_32bit_float(),
                decoder.decode_32bit_float(),
                decoder.decode_32bit_float(),
                decoder.decode_32bit_float(),
                decoder.decode_32bit_float(),
                decoder.decode_32bit_float(),
                decoder.decode_32bit_float(),
                decoder.decode_32bit_float(),
                decoder.decode_32bit_float()
            ]

            # L1 contains [ V1, V2, V0, I1, I2, I0 ]
            # Voltage and current values 
            l1 = decoded_features[:6]

            relay = 'RTL3'
            l11.append(l1)
            print("\n")
            i = i+1
            print(f"Iteration {i} completed")
            print(decoded_features)
            [p_cnn, pf, pfc] = compute_fault_probabilities(l1, l11, relay)

            print(f"Configuration classifier probability (DCNN): {p_cnn}")
            print(f"Fault Detector probability (GP + OCSVM): {pf}")
            print(f"Fault classifier probability (SVM): {pfc}")
            # result1 = client.read_holding_registers()
        except KeyboardInterrupt:
            print('Keyboard Interrupt')
            client.close()
