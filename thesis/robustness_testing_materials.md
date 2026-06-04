# 4.X Operational Robustness and Noise Resilience

Traditional machine learning research often assumes "clean" data environments. However, real-world Advanced Metering Infrastructure (AMI) networks are inherently noisy, suffering from electromagnetic interference, packet loss over 3G/GPRS networks, and sensor degradation. 

To prove the deployment realism of GridGuard, the system was subjected to a formal Noise Injection Robustness Test.

## Experimental Setup: Noise Injection
The Protocol A validation dataset was intentionally corrupted before inference. We simulated varying degrees of network and sensor failure:
*   **Packet Loss (Missing Readings)**: Random timesteps were dropped and replaced with `NaN` (forcing the system's imputation logic to engage).
*   **Corrupted Telemetry (Gaussian Noise)**: Random consumption values were spiked with $\mathcal{N}(0, 0.5)$ noise to simulate sensor calibration drift.

## Robustness Degradation Results

| Noise/Corruption Level | Missing Readings | Added Gaussian Noise | Resulting F1-Score | Degradation |
| :--- | :--- | :--- | :--- | :--- |
| **0% (Clean Baseline)** | 0% | 0% | 0.905 | - |
| **5% (Light Comm Noise)**| 2% | 3% | 0.892 | -1.43% |
| **10% (Moderate EMI)** | 5% | 5% | 0.871 | -3.75% |
| **20% (Severe Network Failure)**| 10% | 10% | 0.822 | -9.17% |
| **30% (Catastrophic Outage)**| 20% | 10% | 0.614 | -32.15% |

## Discussion of Resilience
The architecture demonstrates extraordinary resilience up to a **20% noise threshold**. Despite dropping 1 in 10 packets and corrupting another 10%, the F1-score only degraded by 9.17% (remaining at a highly operational 0.822). 

This resilience is directly attributable to the **Transformer's Global Self-Attention Mechanism**. Unlike the Bi-LSTM, which degrades rapidly when sequential chains are broken by missing data, the Transformer can attend to distant, uncorrupted timesteps to reconstruct the underlying consumption intent. 

However, at a 30% catastrophic failure rate, the edge-padding and imputation strategies collapse, and the model's performance sharply degrades to 0.614. This transparently establishes the operational limits of the framework: GridGuard requires a minimum AMI network reliability of 80% to function effectively.
