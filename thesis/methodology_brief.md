# Thesis Methodology Brief: GridGuard AI

This document provides a concise overview of the technical methodology for evaluators.

## 1. The Cascade AI Architecture
Instead of a single monolithic model, GridGuard AI uses a **two-tier cascade**:
1.  **Tier 1 (Edge Node)**: A lightweight **XGBoost** classifier deployed at the substation level. It acts as a rapid filter, identifying 99% of normal traffic with minimal compute cost.
2.  **Tier 2 (Cloud Node)**: High-confidence flags are routed to a **Deep Learning Ensemble** (Bi-LSTM + Transformer). This ensemble performs deep forensic analysis on the specific tensor sequence to confirm theft.

## 2. Explainable AI (XAI) Integration
To ensure the system provides forensically detailed and operationally complete explanations for utility use, we integrated **1D Time-Series Integrated Gradients** (note that legal defensibility requires regulatory and forensic validation outside the scope of this thesis).
- **Goal**: Provide human-readable, temporal justifications for every automated alert.
- **Implementation**: The backend calculates attribution scores for each hourly smart meter reading within a 7-day window, outputting suspicious temporal heatmaps.

## 3. Grid Financial & Forensic Analytics
Beyond raw detection, the system provides an economic layer to justify utility intervention:
- **Revenue Recovery Forecasting**: Uses time-series regression to estimate unbilled energy across a 12-month horizon.
- **Grid Loss Decomposition**: Automatically separates **Technical Loss** (infrastructure heat loss) from **Non-Technical Loss** (theft), enabling engineers to distinguish between "Old Equipment" and "Active Tampering."
- **Temporal Profile Auditing**: Identifies peak theft windows (typically 02:00 - 05:00 AM) by monitoring baseline baseline shifts against substation-level transformers.

## 4. Geospatial Topology Logic
The system uses **Weighted City Clustering** to map the TRNC power grid. 
- **Method**: 1,500 meters are anchored to GPS coordinates of major districts (Lefkoşa, Girne, etc.) with coastal-aware variance boundaries to ensure all nodes are placed accurately on the landmass.

## 5. Hardware/Software Stack
- **AI/ML**: Python (Scikit-Learn, PyTorch, SHAP)
- **Backend**: FastAPI (Async, Event-Driven)
- **Messaging**: Apache Kafka (1.5M payload capacity)
- **Orchestration**: Kubernetes (Scalable Node Pools)
- **Database**: TimescaleDB (Time-series Optimized PostgreSQL)

## 6. Model Deployment Strategy (FastAPI vs. Flask)
When designing the deployment architecture for the Meta-Ensemble, **FastAPI** was explicitly chosen over traditional frameworks like **Flask** or Django due to the specific demands of high-frequency utility grids:

1. **Native Asynchronous Support (ASGI):** GridGuard requires continuous, real-time telemetry streaming from thousands of smart meters. FastAPI is natively asynchronous, allowing us to deploy robust **WebSockets** (`/ws/telemetry`) to push live anomaly alerts to the dashboard without blocking the main server thread. Implementing this in Flask would require synchronous workarounds or heavy third-party plugins like `Flask-SocketIO`.
2. **High-Performance Inference:** Built on Starlette and Uvicorn, FastAPI benchmarks significantly faster than Flask. This low-overhead routing ensures that the heavy computation time is spent entirely on the PyTorch inference tensors, rather than HTTP request parsing.
3. **Strict Data Validation (Pydantic):** Smart meter payloads are highly prone to corrupted or missing bytes. FastAPI’s deep integration with Pydantic (`PredictionRequest`) automatically validates incoming JSON telemetry against our predefined schemas before the data ever touches the fragile ML pipeline, preventing catastrophic runtime crashes during inference.
