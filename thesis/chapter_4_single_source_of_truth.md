# 4.1 Evaluation Protocols and Primary Metric Framework

Before detailing the granular experimental results, it is critical to establish the dual-evaluation framework utilized to assess the GridGuard Universal Hybrid architecture. 

Evaluating a cyber-physical anomaly detection system solely on a perfectly balanced dataset yields high academic metrics but masks real-world vulnerabilities. Conversely, evaluating solely on a severely skewed production dataset masks the underlying theoretical capacity of the model. Therefore, to ensure both academic rigor and operational realism, this thesis reports results across two distinct evaluative protocols.

## 4.1.1 Protocol A: Controlled Benchmark Evaluation (Primary Academic Metric)

**Protocol A** represents the primary evaluation framework of this thesis. It is utilized to validate **Hypothesis 4 (HA4)**, perform ablation studies, and conduct direct comparative analyses against existing State-of-the-Art (SOTA) literature.

*   **Context**: A 10-Fold Stratified Cross-Validation on a synthetically balanced dataset (using the `TheftInjector` module to ensure a 50/50 class distribution).
*   **Purpose**: To isolate the theoretical discrimination capability of the hybrid ensemble and provide a standardized baseline for algorithmic comparison.
*   **Primary Metrics**:
    *   **F1-Score**: 0.905 (90.5%)
    *   **Precision**: 91.1%
    *   **Recall**: 89.8%
    *   **AUROC**: 0.952
    *   **Accuracy**: 98.23%

*All subsequent academic comparisons, theoretical baselines, and primary hypothesis validations in this chapter refer to Protocol A.*

## 4.1.2 Protocol B: Production-Imbalanced Stress Test

**Protocol B** represents the operational validation of the framework. It assesses the deployability of GridGuard in a realistic utility environment.

*   **Context**: Hold-out test set evaluation on a severely skewed dataset replicating actual KIB-TEK utility conditions (99% normal consumption, 1% electricity theft).
*   **Purpose**: To stress-test the model's threshold robustness and quantify the inevitable degradation of Recall when exposed to overwhelming negative-class noise.
*   **Operational Metrics**:
    *   **F1-Score**: 0.8129 (81.29%)
    *   **Precision**: 92.64%
    *   **Recall**: 72.41%
    *   **AUROC**: 0.9597

While the F1-Score in Protocol B naturally drops compared to Protocol A due to the mathematical realities of extreme class imbalance (the "Accuracy Paradox"), maintaining an F1-Score of 0.8129 under these harsh conditions proves the operational viability of the Meta-Ensemble. 

## 4.1.3 Resolution of Inference Latency Metrics

Because GridGuard is a distributed Edge-Cloud architecture, "inference time" is not a singular metric. To clarify deployment performance, latency is formally decomposed into three distinct operational stages:

1.  **Edge Preliminary Inference ($t_{edge}$): < 1 ms**
    *   The localized execution time of the XGBoost gating model running on the smart meter gateway.
2.  **Cloud Tensor Processing ($t_{cloud}$): 12.25 ms**
    *   The pure PyTorch matrix multiplication time required for the Universal Hybrid (TCN + Bi-LSTM + Transformer) to process the `(1, 26, 2)` tensor inside the Kubernetes ML pod.
3.  **End-to-End API Latency ($t_{api}$): 42.1 ms**
    *   The total real-world operational roundtrip time. This includes Edge-to-Cloud networking, FastAPI JSON serialization, PyTorch tensor routing, Integrated Gradients XAI extraction, and final WebSocket broadcasting to the utility dashboard.
