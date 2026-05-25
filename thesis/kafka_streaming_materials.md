# Thesis Materials: 3.3.13 Real-Time Streaming and Kafka Event Processing Infrastructure

This document connects your advanced deep learning theories to the practical, distributed systems engineering required to deploy GridGuard on a national utility scale. 

## 1. Distributed Event Flow Diagram

This architectural flowchart illustrates the entire lifecycle of a telemetry reading, from the Edge gateway through the Kafka broker, into the ML inference engine, and finally surfacing as an alert via WebSockets.

```mermaid
graph TD
    %% Entities
    subgraph "Edge / Substation Layer"
        Meter[Smart Meter (Raw DNP3)]
        Gateway[Edge Protocol Gateway]
        EdgeFilter[XGBoost Edge Filter]
    end

    subgraph "Kafka Event Streaming (Docker)"
        topic_ingest[(Topic: telemetry.ingest)]
        topic_alerts[(Topic: anomalies.alerts)]
        topic_retrain[(Topic: model.retrain)]
    end

    subgraph "Cloud / Kubernetes Layer"
        FastAPI[FastAPI Backend]
        ML[GridGuard Hybrid Ensemble]
        XAI[Integrated Gradients Engine]
    end

    subgraph "Presentation Layer"
        UI[React Forensic Dashboard]
    end

    %% Data Flow
    Meter -->|15-min Intervals| Gateway
    Gateway -->|Translate to JSON| EdgeFilter
    EdgeFilter -->|Probability > 0.60| topic_ingest
    
    topic_ingest -->|Async Consume| FastAPI
    FastAPI -->|Tensor Routing| ML
    ML -->|Confidence > 0.5270| XAI
    ML -.-> FastAPI
    XAI -.-> FastAPI
    
    FastAPI -->|Publish| topic_alerts
    topic_alerts -->|Stream| FastAPI
    FastAPI -->|WebSocket Broadcast| UI
    
    UI -.->|Operator Marks 'False Positive'| topic_retrain
    topic_retrain -.->|Batch Queue| EdgeFilter
```

## 2. Telemetry Ingestion Payload (JSON Format)

Before telemetry can be processed by your Python ML models, legacy utility protocols (like DNP3 or Modbus) must be standardized. Your `LegacyProtocolGateway` translates these binary blobs into the following standardized JSON payload before publishing to Kafka:

```json
{
    "meter_id": "KIB-TEK-GATEWAY-001",
    "voltage": 231.5,
    "current": 12.4,
    "phase_angle": 120.2,
    "timestamp": 1684930215.123,
    "metadata": {
        "source_protocol": "DNP3",
        "relay_status": "OK"
    }
}
```
*Thesis Note*: Highlight that this decoupling allows GridGuard to remain hardware-agnostic; as long as the edge gateway produces this JSON schema, the cloud ML engine can process it without caring about the physical meter manufacturer.

## 3. Real-Time WebSocket Alerts (FastAPI)

Kafka excels at backend event processing, but browsers cannot natively consume Kafka topics. The FastAPI layer acts as the bridge. 

When the ML Engine confirms an anomaly (publishing to the `anomalies.alerts` topic), the FastAPI server actively consumes that message and instantly broadcasts it to all connected forensic dashboards via WebSockets, ensuring zero-polling latency for utility operators.

```python
# Real-time Telemetry Streaming Route (WebSockets)
@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    try:
        # Internally consumes from 'anomalies.alerts'
        # Emits highly-confident theft detections to the React UI
        while True:
            # Broadcast structure
            payload = {
                "id": f"KIB-TEK-{event['household_id']}",
                "lat": event['lat'],
                "lon": event['lon'],
                "risk": "high",
                "confidence": 0.94,
                "consumption": event['consumption_kwh'],
                "grid_load": event['grid_load_index'],
                "status": "investigating"
            }
            await websocket.send_json(payload)
            await asyncio.sleep(3) # Throttle state updates
    except Exception:
        pass
```

## 4. Retraining Queues & Operator Feedback

A crucial, often-overlooked aspect of AI deployment is handling "Concept Drift" and false alarms. Your architecture implements a specific Kafka workflow for this via the `model.retrain` topic.

From `cloud_node/deep_analysis.py`:

```python
def ingest_false_positive_feedback(self, meter_id, tensor_sequence):
    """
    Receives human-verified 'False Positive' telemetry sequences from the dashboard.
    Initiates a Reinforcement Learning (RL) weight penalty to retrain the Edge Node XGBoost
    during the next off-peak maintenance window to reduce future false alarms.
    """
    logger.info(f"Ingesting FALSE POSITIVE feedback for meter {meter_id}")
    logger.info(f"Scheduling RL weight penalty for sequence profile in Edge Filter retraining queue.")
    
    # Push sequence payload to Kafka topic: model.retrain
    # Nightly batch jobs consume this topic to incrementally tune the model
```

*Thesis Argument*: Emphasize that the utility operator acts as a "Human-in-the-Loop." When an operator inspects the XAI dashboard and marks an alert as a false positive, that exact tensor sequence is immediately pushed to the `model.retrain` Kafka topic. This allows the system to autonomously learn from its mistakes without requiring a full manual dataset rebuild.
