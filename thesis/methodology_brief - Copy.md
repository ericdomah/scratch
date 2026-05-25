# Thesis Methodology Brief: GridGuard AI

This document provides a concise overview of the technical methodology established in the research framework.

## 1. The Cascade AI Architecture
To balance computational efficiency with deep sequential analysis, GridGuard AI utilizes a **two-tier cascade architecture**:
1.  **Tier 1 (Edge Node)**: A lightweight **XGBoost** classifier deployed at the substation level. It acts as a primary statistical filter, identifying 99% of normal operational traffic with an average inference latency of 1.02 ms.
2.  **Tier 2 (Cloud Node)**: Statistically irregular payloads are routed to a **Deep Learning Meta-Ensemble** (TCN + Bi-LSTM + Transformer). This ensemble performs deep sequential forensic analysis, operating at a mean latency of 6.225 ms.

## 2. Explainable AI (XAI) Integration
To ensure the system outputs structured, interpretable explanations for anomaly classifications, the framework integrates **1D Time-Series Integrated Gradients** and SHAP value generation.
- **Objective**: Provide human-readable, temporal justifications for automated alerts to support operational audits.
- **Implementation**: The computational backend calculates attribution scores for each hourly smart meter reading across the sequence window, translating these probabilities into visual temporal heatmaps.

## 3. Grid Analytics and Decomposition
The framework provides supplementary analytical models to contextualize network anomalies:
- **Revenue Recovery Forecasting**: Utilizes time-series regression to estimate potential unbilled energy across evaluated horizons.
- **Grid Loss Decomposition**: Mathematically separates **Technical Loss** (natural infrastructural heat dissipation) from **Non-Technical Loss** (fraudulent diversion), providing diagnostic clarity for distribution engineers.

## 4. Geospatial Topology Simulation
The simulation environment utilizes **Weighted City Clustering** to approximate the topological distribution of the TRNC power grid. 
- **Method**: 1,500 modeled meters are anchored to the spatial coordinates of major regional districts (e.g., Lefkoşa, Girne) utilizing coastal-aware variance boundaries to maintain geographical realism within the simulated dataset.

## 5. Architectural Software Stack
- **AI/ML Engine**: Python, PyTorch, Scikit-Learn, SHAP, XGBoost
- **Backend Service**: FastAPI (Asynchronous ASGI, WebSockets)
- **Data Streaming**: Apache Kafka (Event-Driven Message Brokering)
- **Orchestration**: Docker, Kubernetes (K8s) Cluster Management
- **Persistence**: TimescaleDB (Time-Series Optimized PostgreSQL)

## 6. Deployment Strategy (FastAPI Rationale)
When designing the deployment architecture for the Meta-Ensemble, **FastAPI** was explicitly selected over traditional synchronous frameworks like Flask due to the high-frequency demands of smart grid telemetry:

1. **Native Asynchronous Support (ASGI):** The system requires continuous, real-time telemetry ingestion from smart meters. FastAPI's native asynchronous capabilities facilitate robust **WebSockets** (`/ws/telemetry`) to transmit live anomaly alerts to the operator dashboard without blocking the primary server thread.
2. **Inference Latency Optimization:** Built upon Starlette and Uvicorn, FastAPI provides low-overhead routing, ensuring that computational resources are allocated directly to PyTorch inference tensor operations rather than HTTP request parsing.
3. **Strict Data Validation:** Smart grid telemetry is prone to packet loss and corruption. FastAPI’s native integration with Pydantic (`PredictionRequest`) automatically validates inbound JSON payloads against strict predefined schemas prior to pipeline ingestion, mitigating runtime exceptions during model inference.
