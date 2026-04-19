import torch
import numpy as np
import shap
from model import HybridLSTMTransformer

class XAIEngine:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def get_attention_map(self, x):
        """
        Identifies which time steps the model is focusing on.
        Returns: (seq_len) normalized scores.
        """
        # x shape: (1, seq_len, 1)
        # Extract attention weights from the model
        attn_weights = self.model.get_attention_weights(x)
        
        # Mean across heads and batch, then mean across the query dimension to get token importance
        importance = torch.mean(attn_weights, dim=1).squeeze().cpu().numpy()
        # Normalize to [0, 1] for the heatmap
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
        return importance

    def explain_with_shap(self, x, background_data):
        """
        Uses SHAP to explain feature contribution.
        Note: DeepExplainer is used for PyTorch models.
        """
        explainer = shap.DeepExplainer(self.model, background_data)
        shap_values = explainer.shap_values(x)
        return shap_values

if __name__ == "__main__":
    # Test XAI Engine
    from data_loader import ElectricityDataset
    from torch.utils.data import DataLoader

    print("--- Testing Module 7: Explainable AI Layer ---")
    model = HybridLSTMTransformer()
    xai = XAIEngine(model)
    
    sample_input = torch.randn(1, 20, 1)
    attn_map = xai.get_attention_map(sample_input)
    
    print(f"✓ Attention map extracted. Shape: {attn_map.shape}")
    print(f"✓ Max importance at: {np.argmax(attn_map)}")
    
    assert attn_map.shape == (20,), "Attention map shape mismatch"
    print("\n✅ Module 7 Verification Complete! XAI foundations are in place.")
