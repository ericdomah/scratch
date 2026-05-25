# Individual Model Comparison: Edge vs. Cloud Nodes

GridGuard AI is built upon a two-tier cascade architecture. Rather than relying on a single monolithic system, it distributes the computational workload between two distinct individual models: the **XGBoost Edge Node** and the **Triple-Hybrid Deep Learning Cloud Node**. 

Here is a detailed comparative analysis of the individual models comprising the methodology:

## 📊 Core Model Comparison

| Feature | Tier 1: Edge Node (XGBoost) | Tier 2: Cloud Node (Deep Learning Meta-Ensemble) |
| :--- | :--- | :--- |
| **Primary Role** | High-speed, first-pass statistical filter | Deep sequential forensic analysis |
| **Architecture Type** | Gradient Boosted Decision Trees | TCN + Bi-LSTM + Transformer Encoder |
| **Deployment Location** | Edge Substation / Local Gateway | Centralized Cloud Server / K8s Cluster |
| **Data Perspective** | Tabular / Statistical Aggregations | 1D Sequential Time-Series (26-week window) |
| **Inference Latency** | **1.02 ms** (Highly Optimized) | **6.225 ms** (Computationally Heavier) |
| **Operational Strategy** | Precision-Maximized (Suppresses false positives locally) | Recall-Maximized (Extracts highly disguised theft) |
| **Explainability (XAI)**| Feature Importance (Global distribution limits) | **Integrated Gradients** (Temporal Suspicion Heatmaps) |

---

## 🔍 Technical Trade-offs & Structural Strengths

### 1. XGBoost (The Edge Node)
*   **Strengths:** XGBoost is exceptionally proficient at identifying statistical anomalies in structured data matrices. Optimized for edge-device CPU execution, it filters benign traffic with an inference latency of **1.02 ms**. This rapid execution positions it as an ideal primary layer.
*   **Weaknesses:** The architecture inherently lacks sequential memory capabilities. When processing sophisticated anomalies (e.g., fractional bypasses engineered to mimic seasonal natural variance), XGBoost evaluates discrete tensors without temporal context. Consequently, standard XGBoost baselines exhibit significantly reduced Recall metrics when deployed in isolation.

### 2. Triple-Hybrid Deep Learning (The Cloud Node)
*   **Strengths:** The deep learning ensemble fuses Temporal Convolutional Networks (for immediate structural anomalies), Bi-LSTMs (for maintaining historical progression), and Transformer Encoders (for mapping long-range seasonal periodicities). This architecture comprehends the complete temporal signature of the grid, allowing the identification of highly disguised "Partial Bypasses." Additionally, it natively supports **1D Time-Series Integrated Gradients**, generating specific temporal attributions for anomalies.
*   **Weaknesses:** The deep learning architecture imposes a substantial computational burden. Evaluating complex neural tensors for every smart meter continuously across an entire distribution network introduces severe latency and infrastructure cost constraints. 

## 🧠 Justification for the Cascade Design

The comparative limitations of these individual models provide the structural justification for the proposed two-tier architecture. A system relying exclusively on XGBoost fails to detect sophisticated, chronologically distributed anomalies. Conversely, a network relying entirely on monolithic Deep Learning processing requires financially prohibitive computational overhead.

The **GridGuard Meta-Ensemble** synthesizes these constraints by utilizing the XGBoost model at the edge to efficiently clear 99% of statistically normal telemetry. This routing architecture reserves the computationally intensive Deep Learning forensic analysis strictly for irregular payloads, optimizing both latency and analytical accuracy.
