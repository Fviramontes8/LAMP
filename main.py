import numpy as np
import time

from pymodbus.client.sync import ModbusTcpClient
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadDecoder

import plot_utilities.plot_utils as pu
from model_inference import fault_detector


def main() -> None:
    ip_address = "192.168.1.19"
    port = 502
    client = ModbusTcpClient(ip_address, port)
    print(client.connect())
    print(client.is_socket_open())
    total_data = []
    iteration = 0

    decoded_feature_labels = [
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

    p_cnn_history = np.zeros((1, 4))
    pf_history = []
    pfc_history = np.zeros((1, 3))

    while client.connect():
        time.sleep(1)
        try:
            result = client.read_holding_registers(0, 26, unit=1)
            decoder = BinaryPayloadDecoder.fromRegisters(
                result.registers,
                byteorder=Endian.Little,
                wordorder=Endian.Little
            )

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
            total_data.append(l1)
            p_cnn, pf, pfc = fault_detector.compute_fault_probabilities(
                l1,
                total_data,
                relay
            )

            print(f"Iteration: {iteration} completed\n")
            # print(f"Iteration {iteration}, p_cnn: {p_cnn}, pf: {pf}, pfc: {pfc}")

            p_cnn_history = np.vstack((p_cnn_history, p_cnn))
            pf_history.append(pf[0])
            pfc_history = np.vstack((pfc_history, pfc))
            pu.save_cnn_history(p_cnn_history)
            pu.save_pf_history(pf_history)
            pu.save_pfc_history(pfc_history)

            iteration = iteration + 1
        except KeyboardInterrupt:
            print('Keyboard Interrupt')
            client.close()


if __name__ == "__main__":
    main()
