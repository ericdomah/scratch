import torch
import numpy as np
import shap

class XAIEngine:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def get_saliency_map(self, x):
        """
        Calculates Gradient Saliency to identify which days influenced the decision.
        x shape: (1, seq_len, 1)
        Returns: (seq_len) normalized importance scores.
        """
        # Ensure input requires gradient
        x = x.to(self.device).detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(x)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass (gradient of the output with respect to input)
        output.backward()
        
        # Saliency is the absolute value of the gradient
        # x.grad shape: (1, seq_len, 1)
        saliency = x.grad.data.abs().squeeze()
        
        # Normalize to [0, 1]
        if saliency.max() > 0:
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency.cpu().numpy()

    def get_integrated_gradients(self, x, baseline=None, steps=50):
        """
        More robust attribution method: Integrated Gradients.
        """
        if baseline is None:
            baseline = torch.zeros_like(x).to(self.device)
        
        x = x.to(self.device)
        scaled_inputs = [baseline + (float(i) / steps) * (x - baseline) for i in range(0, steps + 1)]
        
        grads = []
        for s_in in scaled_inputs:
            s_in = s_in.detach().requires_grad_(True)
            out = self.model(s_in)
            self.model.zero_grad()
            out.backward()
            grads.append(s_in.grad.data)
            
        avg_grads = torch.mean(torch.stack(grads), dim=0)
        integrated_grad = (x - baseline) * avg_grads
        
        importance = integrated_grad.abs().squeeze()
        if importance.max() > 0:
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
            
        return importance.cpu().numpy()

if __name__ == "__main__":
    from ensemble_model import GridGuardUniversalHybrid
    print("--- Verifying XAI Engine (Gradient Saliency) ---")
    model = GridGuardUniversalHybrid()
    xai = XAIEngine(model)
    
    sample_input = torch.randn(1, 30, 1)
    saliency = xai.get_saliency_map(sample_input)
    
    print(f"[OK] Saliency map extracted. Shape: {saliency.shape}")
    print(f"[OK] Peak suspicion at day: {np.argmax(saliency)}")
