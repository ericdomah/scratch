# Chapter X: Evaluation & Experimental Results

To validate the efficacy of the proposed **GridGuard AI Hybrid Learning Architecture**, a comprehensive comparative study was conducted. The experimental setup tests the performance of structural and ensemble models against highly imbalanced real-world electricity consumption data spanning the SGCC open-source datasets.

## 1. Experimental Architectures Assessed
The study empirically bounds the problem of Non-Technical Losses (NTL) using three distinct algorithmic layers, subsequently fusing them into a final probabilistic voting mechanism:

1. **XGBoost (Baseline)**: A highly-efficient gradient boosting tree mathematically flattened to process seq-to-seq data, providing a robust statistical baseline.
2. **Hybrid LSTM-Transformer (Core Model)**: A deep learning network featuring an `nn.LSTM` extraction layer to handle contiguous temporal logic, fed into an `nn.TransformerEncoder` network to parse multidimensional seasonal variance representations.
3. **Temporal Fusion Transformer (TFT)**: An attention-native sequence architecture emphasizing intrinsic explainability and continuous forecasting stability.
4. **Meta-Ensemble (Proposed Hybrid Protocol)**: A soft-voting classifier deriving probabilities from the three foundational models, averaging out algorithmic blindspots to prioritize maximal general feature capture and stabilize Precision-Recall differentials in severely imbalanced subsets.

## 2. Training Convergence
Execution was logged using Cross-Entropy bounds for the dense gradient steps. The learning trajectories confirm structural fitting bounds within minimal epoch limits.

![Training Loss Curve](file:///C:/Users/eric.domah/.gemini/antigravity/scratch/ml_engine/src/outputs/training_loss_curve.png)

## 3. Confusion Matrices
To dissect false-positive vs. false-negative penalties, confusion matrices directly illustrate the misclassification volume.

![Confusion Matrices](file:///C:/Users/eric.domah/.gemini/antigravity/scratch/ml_engine/src/outputs/confusion_matrices.png)

*Figure X.1: Quadrant view mapping True Normal, True Theft, Flase Positives, and False Negatives across the specific algorithms.*

## 4. ROC Area Under Curve (AUC) & Probability Densities
The ROC topology asserts model reliability irrespective of specific classification thresholds. The Ensembled framework routinely asserts higher continuous dominance in the AUC plane over strict tree baselines by mitigating edge-case variance seen strictly in the DL subset paths.

![ROC Curves](file:///C:/Users/eric.domah/.gemini/antigravity/scratch/ml_engine/src/outputs/roc_curve_comparison.png)

## 5. Precision-Recall Stability (Critical Metric)
Because electricity theft represents a microscopic minority percentage of the massive consumer grid, traditional accuracy and basic ROC figures can mathematically inflate success context. **Precision-Recall (PR) curves** are the definitive academic validation mechanism for this class.

![Precision-Recall Curves](file:///C:/Users/eric.domah/.gemini/antigravity/scratch/ml_engine/src/outputs/pr_curve_comparison.png)

## 6. Conclusion
The implementation of a `Meta-Ensemble` voting system over structural LSTMs, Attention blocks, and Tree gradients demonstrably improves feature retention and bounds stability. Real-world SCADA integrations (as prototyped in the GridGuard UI) utilizing this analytical layer can successfully reduce false operator dispatches while trapping sophisticated synthetic bypass vectors.
