# 4.X Threats to Validity

To ensure scientific honesty and frame the limits of this research, the following threats to validity are formally acknowledged. 

## 1. Internal Validity (In-Sample Bias)
The most significant threat to internal validity is the reliance on the synthetic `TheftInjector` module. Because real, labeled theft data from KIB-TEK was legally and operationally unavailable, the system was trained to detect five specific mathematical models of theft (e.g., partial bypass, constant reduction). If real-world utility thieves employ entirely novel physical bypass methods that do not align with these five synthetic topologies, the model’s true Recall will be lower than the reported 89.8%.

## 2. External Validity (Deployment Reality)
While the model proved mathematically viable during the cross-domain SGCC transfer test, it has not yet been deployed in a live, streaming KIB-TEK substation. The real-world constraints of actual smart meter hardware—such as 3G network latency, MQTT broker timeouts, or adversarial physical tampering of the edge gateway—could introduce systemic failures not captured by the current Kubernetes simulation.

## 3. Construct Validity (The GLI Proxy)
The GridGuard architecture relies heavily on the Grid Load Index (GLI) as a contextual regularizer. However, the true "Grid Load" is a complex, non-linear function of reactive power, voltage drops, and transformer heat. This thesis utilized a simplified, normalized proxy for GLI. If the real distribution transformers exhibit drastically different phase-imbalance profiles, the Transformer’s contextual attention mechanism may lose its stabilizing effect.

## 4. Conclusion Validity (Statistical Power)
While multi-seed testing (10 seeds) and 10-Fold Cross-Validation were utilized to calculate confidence intervals, the overall dataset size (roughly 100,000 samples) is relatively small for deep learning standards. A larger dataset spanning multiple years of consumption (to capture extreme weather anomalies like 100-year heatwaves) would be required to definitively confirm the statistical bounds of the F1-score.
