# 4.X Reproducibility Statement

The credibility of computational research hinges entirely on its reproducibility. To ensure that future researchers and utility engineers can independently verify the results presented in this thesis, a strict open-science reproducibility framework was established.

## Repository and Codebase Access
The complete, end-to-end GridGuard architecture—including the Edge filter, Cloud DL pipeline, FastAPI backend, and XAI dashboard—is version-controlled and maintained in a Git repository.

*   **Environment Standardization**: All dependencies are strictly locked via a provided `requirements.txt` and `Dockerfile`, ensuring the PyTorch 2.1.0+cu118 and XGBoost 2.0.2 environments compile identically across all operating systems.
*   **Seed Documentation**: To combat stochastic variance in deep learning optimization, the global pseudo-random number generator (PRNG) seed used for the primary Protocol A benchmark is hard-coded as `SEED = 42` across the `torch`, `numpy`, and `random` libraries.
*   **Data Generation Pipeline**: Because actual KIB-TEK utility data is protected under severe privacy regulations and cannot be open-sourced, the repository includes the `TheftInjector` and `ElectricityDataset` modules. Executing `python data_pipeline.py` will autonomously synthesize the exact (98,412 Train / 10,935 Validation) dataset tensor geometries utilized in this study, allowing researchers to replicate the exact training environment without requiring legally restricted telemetry.

By adhering to these open-science principles, this thesis transcends being a closed-box engineering report and provides a verifiable, extensible baseline for future Advanced Metering Infrastructure (AMI) research.
