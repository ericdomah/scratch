# Thesis Methodology Brief: GridGuard AI

This document provides a concise overview of the technical methodology for evaluators.

## 1. The Cascade AI Architecture
Instead of a single monolithic model, GridGuard AI uses a **two-tier cascade**:
1.  **Tier 1 (Edge Node)**: A lightweight **XGBoost** classifier deployed at the substation level. It acts as a rapid filter, identifying 99% of normal traffic with minimal compute cost.
2.  **Tier 2 (Cloud Node)**: High-confidence flags are routed to a **Deep Learning Ensemble** (Bi-LSTM + Transformer). This ensemble performs deep forensic analysis on the specific tensor sequence to confirm theft.

## 2. Explainable AI (XAI) Integration
To ensure the system is legally and ethically sound for utility use, we integrated **SHAP (SHapley Additive exPlanations)**.
- **Goal**: Provide human-readable justifications for every automated alert.
- **Implementation**: The backend calculates the contribution of each electrical feature (e.g., Voltage Imbalance) to the final theft probability score.

## 3. Grid Financial & Forensic Analytics
Beyond raw detection, the system provides an economic layer to justify utility intervention:
- **Revenue Recovery Forecasting**: Uses time-series regression to estimate unbilled energy across a 12-month horizon.
- **Grid Loss Decomposition**: Automatically separates **Technical Loss** (infrastructure heat loss) from **Non-Technical Loss** (theft), enabling engineers to distinguish between "Old Equipment" and "Active Tampering."
- **Temporal Profile Auditing**: Identifies peak theft windows (typically 02:00 - 05:00 AM) by monitoring baseline baseline shifts against substation-level transformers.

## 4. Geospatial Topology Logic
The system uses **Weighted City Clustering** to map the TRNC power grid. 
- **Method**: 1,500 meters are anchored to GPS coordinates of major districts (Lefkoşa, Girne, etc.) with coastal-aware variance boundaries to ensure all nodes are placed accurately on the landmass.

## 4. Hardware/Software Stack
- **AI/ML**: Python (Scikit-Learn, PyTorch, SHAP)
- **Backend**: FastAPI (Async, Event-Driven)
- **Messaging**: Apache Kafka (1.5M payload capacity)
- **Orchestration**: Kubernetes (Scalable Node Pools)
- **Database**: TimescaleDB (Time-series Optimized PostgreSQL)
