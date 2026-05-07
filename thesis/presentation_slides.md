# GridGuard AI: Defense Presentation Slides

*Note: This is a structured outline for your final PowerPoint/Keynote presentation. Each "Slide" represents a core visual or talking point.*

---

## Slide 1: Title Slide
*   **Title:** GridGuard AI for Power Theft Detection for Smart Grids
*   **Subtitle:** Securing the TRNC Smart Grid with Explainable Deep Learning
*   **Visual:** The GridGuard AI dashboard logo or a sleek map of the TRNC grid.
*   **Footer:** Your Name | Defense Date | Thesis Committee

## Slide 2: The Problem (The False Positive Crisis)
*   **Headline:** Current SOTA models flag too many normal customers.
*   **Bullet Points:**
    *   Electricity theft (Non-Technical Loss) costs millions annually.
    *   Standard models (Vanilla LSTM) achieve high detection rates but fail in practice due to **False Positive Fatigue**.
    *   *Example:* A family going on holiday creates a sudden drop in consumption, which standard AI incorrectly flags as a bypass.
*   **Visual:** A graph showing a normal drop vs. an actual theft drop.

## Slide 3: The Solution (Context-Aware Intelligence)
*   **Headline:** The Grid Load Index
*   **Bullet Points:**
    *   Our model doesn't just look at the individual meter; it looks at the surrounding **transformer load**.
    *   If a house's consumption drops, but the neighborhood's demand is high, it is highly suspicious.
    *   This simple but profound contextual correlation boosts Precision to **90.6%**.

## Slide 4: The Architecture (Two-Tier Cascade)
*   **Headline:** Edge-to-Cloud Meta-Ensemble
*   **Bullet Points:**
    *   Running deep learning on millions of meters 24/7 is financially impossible.
    *   **Tier 1 (Edge Node):** XGBoost processes tabular data in 0.003ms. Clears 99% of normal traffic.
    *   **Tier 2 (Cloud Node):** The Super-Hybrid (Bi-LSTM + Transformer) runs deep forensic sequence analysis *only* on the flagged 1%.
*   **Visual:** A flowchart showing data moving from Smart Meter -> Substation (XGBoost) -> Cloud Server (Transformer).

## Slide 5: Data Augmentation (Digital Twin)
*   **Headline:** Physics-Grounded Theft Injection
*   **Bullet Points:**
    *   Real theft data is protected by utility privacy laws.
    *   Instead of standard mathematical oversampling (SMOTE), we built a **Digital Twin**.
    *   We programmatically synthesize realistic hardware tampering (e.g., 30% phase bypasses during off-peak hours) into normal sequences.

## Slide 6: Legal & Ethical Viability (Explainable AI)
*   **Headline:** Moving beyond the Black Box
*   **Bullet Points:**
    *   Utility companies cannot legally disconnect a customer based solely on an "AI Score."
    *   GridGuard integrates **1D Time-Series Integrated Gradients (XAI)**.
    *   Generates a temporal heatmap, allowing human engineers to pinpoint the exact day/hour the meter was bypassed.
*   **Visual:** A screenshot of your `xai_report.png` showing the red suspicious zones over the 30-day tensor.

## Slide 7: Results & Benchmarking
*   **Headline:** Superiority over Baseline Models
*   **Table:**
    *   *Industry Baseline (XGBoost):* Recall 2.0% | F1 0.04
    *   *Academic Baseline (Vanilla LSTM):* Precision 8.1% | F1 0.15
    *   **GridGuard Context-Aware:** Recall 100% | Precision 90.6% | **F1 0.95**
*   **Visual:** The comparative ROC and Precision-Recall Curves (from the `outputs` folder).

## Slide 8: Live Demonstration
*   **Headline:** Real-Time Dashboard (FastAPI + React)
*   **Bullet Points:**
    *   Asynchronous telemetry streaming via WebSockets.
    *   Geospatial clustering of high-risk regions.
    *   One-click **Forensic Audit PDF Export**.
*   **Visual:** "Switching to Live Demo..."

## Slide 9: Conclusion & Future Work
*   **Headline:** A production-ready blueprint for national deployment.
*   **Bullet Points:**
    *   Successfully solved the false positive crisis.
    *   Provided legally defensible XAI.
    *   **Future Work:** Edge model distillation (running the model entirely on the meter hardware) and GDPR-compliant federated learning.

## Slide 10: Q&A
*   **Headline:** Questions?
*   **Visual:** A clean "Thank You" slide with your contact info or GitHub link.
