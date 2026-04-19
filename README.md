# GridGuard AI: Explainable Electricity Theft Detection System

GridGuard AI is an end-to-end system for detecting electricity theft using long short-term memory (LSTM) and Transformer models, with a focus on explainability (XAI) using Attention and SHAP.

## 🏗️ System Architecture
- **Data Layer**: Real-world dataset handling (SGCC focus).
- **ML Engine**: PyTorch-based LSTM+Transformer hybrid.
- **Backend**: FastAPI for inference and data serving.
- **Frontend**: React + Shadcn/UI for monitoring and XAI heatmaps.
- **Deployment**: Docker Compose.

## 📁 Directory Structure
- `/backend`: FastAPI source code.
- `/frontend`: React dashboard source code.
- `/ml_engine`: Core model architecture and training scripts.
- `/data`: Raw and processed dataset files.
- `/notebooks`: Exploratory Data Analysis (EDA) and experimental model training.

## 🚀 Getting Started
(Detailed setup instructions will be added as we build each module.)
