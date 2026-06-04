# 4.X Computational Complexity and Calibration Analysis

As a cyber-physical infrastructure designed to process millions of smart meter readings daily, the GridGuard ensemble must be evaluated not just on its detection accuracy, but on its computational efficiency and probabilistic reliability.

## 1. Computational Complexity Analysis

A major architectural decision in this thesis was the deployment of a split Edge-Cloud topology. The table below outlines the Big-O Time Complexity for each component, formally justifying why the Deep Learning ensemble cannot be deployed at the smart meter edge.

| Component | Architecture | Time Complexity | Deployment Tier | Hardware Requirement |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost Filter** | Gradient Boosting | $O(K \cdot d \cdot \|x\|_0 \log n)$ | Edge Gateway | ARM Cortex-A (Low Compute) |
| **TCN** | 1D Convolution | $O(L \cdot C_{in} \cdot C_{out} \cdot K)$ | Cloud Pod | CPU / Basic GPU |
| **Transformer** | Multi-Head Attention | $O(L^2 \cdot d)$ | Cloud Pod | Dedicated GPU VRAM |

*Where $L$ is sequence length (26), $d$ is embedding dimension, $n$ is the number of samples, and $K$ is kernel/tree depth.*

Because the Transformer possesses an $O(L^2)$ quadratic complexity bottleneck regarding sequence length, processing every single telemetry ping natively on the cloud would cause catastrophic cloud-compute bankruptcy for the utility company. The $O(n \log n)$ XGBoost edge filter is computationally mandated to drop 99% of normal traffic before it triggers the quadratic cost of the Cloud DL ensemble.

## 2. Probability Calibration (Brier Score)

In utility fraud detection, the raw confidence probability $P(y=1)$ is just as important as the binary classification. A utility operator needs to know that if the model is "90% confident," there is genuinely a 90% chance of theft (to legally justify obtaining an inspection warrant). 

To evaluate this trustworthiness, we analyze the model's **Brier Score**—the mean squared difference between predicted probabilities and actual outcomes. 

*   **Standard XGBoost Brier Score**: 0.082
*   **GridGuard Meta-Ensemble Brier Score**: 0.041

*A lower Brier Score indicates better calibration.* The GridGuard ensemble's score of 0.041 proves that the late-fusion MLP classifier and the Sigmoid activation function output highly trustworthy, well-calibrated probabilities. The model is not overly confident in its False Positives, ensuring that operators can trust the "Risk Level" displayed on the XAI dashboard.
