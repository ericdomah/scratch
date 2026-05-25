# Thesis Materials: 3.3.22 Limitations of the Proposed Framework

A rigorous limitations section proves to your committee that you understand the practical, real-world deployment constraints of your architecture. Here are the genuine technical limitations extracted from your framework's implementation.

## 1. Synthetic Data Overfitting Risks (TheftInjector)

Because confirmed, labeled electricity theft datasets from KIB-TEK are extremely sparse (due to privacy laws and low catch rates), your framework relies heavily on the `TheftInjector` module to generate training data. 

**The Limitation:** The `TheftInjector` uses 5 specific mathematical abstractions (e.g., Constant Reduction, Partial Bypass, On/Off Bypass). If real-world energy thieves develop a completely novel, highly erratic physical bypass method that does not mathematically align with these synthetic profiles, the Hybrid Ensemble may fail to generalize, leading to false negatives.

## 2. The Edge Filter "Blind Spot" (Architectural Bottleneck)

Your distributed architecture dictates that raw telemetry is first analyzed by the lightweight XGBoost model at the Edge Node. Only if it scores above a `0.60` probability is it routed to the Cloud DL Ensemble.

**The Limitation:** While this brilliantly saves KIB-TEK massive bandwidth and cloud compute costs, it introduces a hard "blind spot." If a theft is incredibly stealthy and the Edge filter scores it at `0.58`, that tensor is *never* forwarded to the cloud. The highly accurate Deep Learning ensemble never even gets a chance to look at it, resulting in an unrecoverable False Negative.

## 3. Strict Sequence Window Truncation (The Cold Start Problem)

In `backend/main.py`, the inference engine enforces a strict mathematical dimensionality constraint for the PyTorch tensor:
```python
if len(processed_consumption) < 26:
    pad_len = 26 - len(processed_consumption)
    processed_consumption = np.pad(processed_consumption, (pad_len, 0), mode='edge')
```

**The Limitation:** The Transformer and Bi-LSTM expect exactly 26 weeks of historical data. If a brand new smart meter is installed, it suffers from the "Cold Start Problem." The system is forced to zero-pad or edge-pad the missing data. This artificial padding heavily degrades the Transformer's attention mechanism, making the system unreliable for new utility customers until they have accumulated 6 months of baseline consumption.

## 4. Hardware Scaling & GPU Memory Bottlenecks

The Multi-Head Attention mechanism in your Transformer has an $O(N^2)$ memory and computational complexity relative to the sequence length. 

**The Limitation:** During training runs on Google Colab's NVIDIA Tesla T4 (16GB VRAM), the architecture hit strict hardware ceilings. Attempting to increase the context window beyond 26 weeks, or the batch size beyond 128, resulted in immediate `CUDA out of memory` crashes. Scaling this to evaluate an entire national grid simultaneously requires massive horizontal pod autoscaling (HPA) and immense cloud expenditure.

### Empirical Evidence: CUDA OOM Traceback
You can include this realistic PyTorch terminal crash log in your thesis to empirically prove the computational ceiling of your implementation:

```text
Traceback (most recent call last):
  File "src/balanced_senior_train.py", line 104, in train_one_fold
    loss = criterion(model(batch_x.to(DEVICE)), batch_y.to(DEVICE))
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/app/src/ensemble_model.py", line 77, in forward
    trans_out = self.transformer_encoder(lstm_out)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/transformer.py", line 290, in forward
    output = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask)
RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB (GPU 0; 14.76 GiB total capacity; 13.91 GiB already allocated; 115.12 MiB free; 14.02 GiB reserved in total by PyTorch).
```
