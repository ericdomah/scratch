import urllib.request
import base64
import os

mermaid_code = """graph TD
    %% Data Sources
    A1((Smart Meters)) -->|15-min Telemetry| B
    A2((Distribution Transformer)) -->|Aggregated Load| B
    
    subgraph "Grid Edge (Substation)"
        B[Data Ingestion & Preprocessing] --> C
        C[Context-Aware Intelligence Layer] -->|kWh + GLI Tensor| D
        D{XGBoost Statistical Gatekeeper}
        D -->|Normal (99%)| E[Local Storage]
        D -->|Anomalous (1%)| F((Cloud Transmission))
    end
    
    subgraph "Central Cloud (Forensic Analysis)"
        F --> G[GridGuard Meta-Ensemble]
        G --> H1[TCN: Local Features]
        G --> H2[Bi-LSTM: Sequential Dependencies]
        H1 --> I[Transformer Encoder: Global Attention]
        H2 --> I
        I --> J[Fully Connected Layer]
        J --> K{Threshold & Classification}
    end
    
    subgraph "Utility SOC Dashboard"
        K -->|Theft Prob > 0.60| L[XAI Engine: Temporal Heatmap]
        K -->|Benign| M[Grid Loss Accounting]
        L --> N((Field Technician Alert))
    end

    classDef edge fill:#f9f2ec,stroke:#d9b38c,stroke-width:2px;
    classDef cloud fill:#e6f3ff,stroke:#99ccff,stroke-width:2px;
    classDef dash fill:#e6ffe6,stroke:#99ff99,stroke-width:2px;
    
    class B,C,D edge;
    class G,H1,H2,I,J,K cloud;
    class L,M dash;
"""

encoded_string = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
url = f"https://mermaid.ink/img/{encoded_string}?bgColor=white"

output_path = "C:/Users/User/Downloads/scratch-main/thesis/images/architecture_diagram.png"

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
try:
    with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
        out_file.write(response.read())
    print("Successfully downloaded architecture_diagram.png")
except Exception as e:
    print(f"Failed to download: {e}")
