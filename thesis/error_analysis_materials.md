# 4.X Qualitative Error Analysis (False Positives & False Negatives)

While aggregate metrics like F1-score provide a macroscopic view of model performance, a responsible cyber-physical deployment requires a microscopic understanding of model failures. In the high-stakes domain of utility revenue protection, a False Positive leads to a costly and legally hostile physical inspection, while a False Negative leads to continuous revenue hemorrhage.

To improve operational maturity, a qualitative analysis was conducted on the test set predictions to understand *why* the GridGuard Meta-Ensemble failed when it did.

## Analysis of False Positives (False Accusations)
A False Positive occurs when the system flags a legitimate consumer as a thief. Upon reviewing the XAI dashboard outputs for these errors, two dominant patterns emerged:

1.  **Seasonal Tourism Spikes (The "Villa" Effect)**:
    In the TRNC dataset, many coastal properties remain completely dormant (near-zero consumption) during the winter, followed by a massive, sustained spike in July and August. If this spike coincides with a localized drop in the Grid Load Index (GLI), the Transformer's attention mechanism occasionally misinterprets the sudden volatility as an "On/Off Bypass," triggering a false alarm.
2.  **Unregistered Solar Photovoltaic (PV) Feed-in**:
    Consumers who installed rooftop solar panels without properly registering for net-metering exhibited sudden, deep drops in grid consumption during sunny hours. The TCN recognized this as a "Partial Bypass" theft signature, accurately detecting the anomaly but misclassifying the physical intent.

## Analysis of False Negatives (Missed Theft)
A False Negative occurs when a thief successfully evades detection. The evaluation revealed that GridGuard is highly susceptible to one specific attack vector:

1.  **Ultra-Stealthy Gradual Manipulation (The "Frog Boiling" Attack)**:
    When the synthetic `TheftInjector` applied a constant reduction attack that mathematically drifted downwards by only 1% to 2% per month over a 6-month window, the system failed to trigger. Because the TCN relies on sudden localized drops, and the Bi-LSTM incorporates the slow drift into its "new normal" memory state, the ensemble confidence never breached the `0.5270` decision threshold. This indicates that highly sophisticated, digitally adaptive attackers could theoretically evade the current architecture by manipulating smart meter firmware to throttle theft gradually over multiple seasons.
