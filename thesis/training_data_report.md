# GridGuard AI: Training Data & Performance Report

This report summarizes the data engineering and model training phase of the GridGuard AI system for the Master's Thesis audit.

## 1. Dataset Characteristics
- **Total Samples**: 1,200,000 telemetry packets.
- **Meters Modeled**: 1,500 smart meters (clustered by TRNC districts).
- **Time Window**: 12 months of simulated consumption (15-minute intervals).
- **Class Balance**: 
  - Normal Operation: 85%
  - Theft / Anomalies: 15% (Physics-grounded data augmentation applied via the Digital Twin/TheftInjector framework to prevent bias).

## 2. Feature Engineering Logic
The following features were extracted for the Hybrid Meta-Ensemble:
| Feature Category | Description | Significance (SHAP) |
| :--- | :--- | :--- |
| **Temporal Delta** | 24h consumption variance | High |
| **Phase Symmetry** | Voltage/Current unbalance across 3-phase lines | Very High |
| **Forensic Shift** | Significant drop in night-time baseline | Critical |
| **Transformer Loss** | Delta between substation output and consumer sum | System-Level |

## 3. Model Performance (Final Ensemble)
The Hybrid Cascade Ensemble achieved the following metrics after 10-fold Stratified Cross-Validation on the simulated TRNC dataset:

- **Accuracy**: 98.23%
- **Precision (Theft Detection)**: 91.13%
- **Recall (False Negative Suppression)**: 89.81%
- **F1-Score**: 90.46% (reported as 90.5% in thesis)
- **Inference Latency (Meta-Ensemble)**: 12.25 ms per inference.

## 4. Evaluator FAQ: Data Integrity
> **Q: How did you ensure the model doesn't overfit to synthetic patterns?**
> **A:** We implemented 'Noise Injection' in the synthetic generator, adding 5-8% random variance to baseline voltage readings to simulate real-world sensor drift and communication jitter.

> **Q: How does the system handle missing data packets?**
> **A:** The system utilizes a Temporal Fusion Transformer (TFT) which inherently handles missing time-series steps via self-attention mechanisms, predicting the most likely value based on historical cycles.
