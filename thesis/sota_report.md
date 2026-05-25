# SOTA Comparative Evaluation Report

This report benchmarks the **GridGuard AI Meta-Ensemble** against academic and industrial baselines.

| Model | Recall | Precision | F1 | AUROC | Inference (ms) | XAI |
| --- | --- | --- | --- | --- | --- | --- |
| Vanilla LSTM (2019 Baseline) | 1.00000 | 0.14500 | 0.25328 | 0.37594 | 0.047623 | No |
| Standard XGBoost (Utility Std) | 0.54023 | 0.95918 | 0.69118 | 0.93527 | 0.035607 | Limited |
| GridGuard Meta-Ensemble (Ours) | 0.72414 | 0.92647 | 0.81290 | 0.95971 | 0.456058 | Yes (Integrated Gradients) |


### Thesis Defense Insights:
- **Methodology Superiority:** GridGuard achieves higher Recall than standard XGBoost by leveraging sequential memory.
- **Operational Efficiency:** Despite being a complex ensemble, inference latency remains under 1ms per meter.
- **Transparency:** Only GridGuard provides native 1D Time-Series XAI support.
