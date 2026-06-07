"""
autoencoder.py
--------------
Autoencoder-based anomaly detector for theft detection.

Components
----------
Encoder
    FC layers with BatchNorm, ReLU, and Dropout that compress the input
    into a low-dimensional latent representation.

Decoder
    Mirror of the encoder; reconstructs the input from the latent code.

TheftDetector
    High-level wrapper that:
        * fit(X_normal)          - trains the autoencoder on normal samples
        * predict_proba(X)       - returns normalised reconstruction error ∈ [0, 1]
        * set_threshold()        - picks the anomaly threshold via a percentile
        * predict(X)             - returns binary labels {0, 1}

The reconstruction error (MSE per sample) serves as the anomaly score;
high error → likely anomalous (theft).  The score is min-max normalised
using statistics collected during fit() to produce comparable probabilities.

Note
----
TheftDetector.predict_proba() returns a (B, 1) tensor compatible with the
binary classification pipeline used by the other models in this package.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Fully-connected encoder with BatchNorm, ReLU, and Dropout.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the (flattened) input.
    hidden_dims : List[int]
        Sizes of the hidden layers.  The last element is the latent dimension.
    dropout : float
        Dropout probability applied after each hidden activation. Default ``0.2``.
    use_batch_norm : bool
        Apply BatchNorm1d before activation at each layer. Default ``True``.

    Examples
    --------
    >>> enc = Encoder(input_dim=64, hidden_dims=[32, 16])
    >>> z = enc(torch.randn(8, 64))
    >>> z.shape
    torch.Size([8, 16])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one element.")

        layers: List[nn.Module] = []
        in_dim = input_dim
        for out_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(in_dim, out_dim, bias=not use_batch_norm))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            in_dim = out_dim

        # Latent layer — no BN/Dropout to keep latent space clean
        layers.append(nn.Linear(in_dim, hidden_dims[-1]))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Flattened input ``(B, input_dim)``.

        Returns
        -------
        torch.Tensor
            Latent code ``(B, hidden_dims[-1])``.
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """Fully-connected decoder that mirrors the encoder.

    Parameters
    ----------
    latent_dim : int
        Size of the latent code produced by the encoder.
    hidden_dims : List[int]
        Reverse hidden layer sizes (mirror of encoder), **excluding** the
        final output layer which reconstructs to ``output_dim``.
    output_dim : int
        Dimensionality of the reconstructed output.
    dropout : float
        Dropout probability. Default ``0.2``.
    use_batch_norm : bool
        Apply BatchNorm1d at hidden layers. Default ``True``.

    Examples
    --------
    >>> dec = Decoder(latent_dim=16, hidden_dims=[32], output_dim=64)
    >>> x_hat = dec(torch.randn(8, 16))
    >>> x_hat.shape
    torch.Size([8, 64])
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        in_dim = latent_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim, bias=not use_batch_norm))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            in_dim = out_dim

        # Final reconstruction layer
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to reconstruction.

        Parameters
        ----------
        z : torch.Tensor
            Latent code ``(B, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstructed input ``(B, output_dim)``.
        """
        return self.net(z)


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    """Encoder + Decoder autoencoder module.

    Parameters
    ----------
    input_dim : int
        Flattened input dimensionality.
    hidden_dims : List[int]
        Encoder hidden + latent dimensions.  The decoder mirrors these in
        reverse (excluding the latent dim which becomes the decoder input).
    dropout : float
        Dropout probability throughout. Default ``0.2``.
    use_batch_norm : bool
        Toggle BatchNorm. Default ``True``.

    Examples
    --------
    >>> ae = Autoencoder(input_dim=64, hidden_dims=[32, 16])
    >>> x = torch.randn(8, 64)
    >>> x_hat = ae(x)
    >>> x_hat.shape
    torch.Size([8, 64])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

        latent_dim = hidden_dims[-1]
        decoder_hidden = list(reversed(hidden_dims[:-1]))  # mirror

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden,
            output_dim=input_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, input_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstruction ``(B, input_dim)``.
        """
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent code.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, input_dim)``.

        Returns
        -------
        torch.Tensor
            Latent code ``(B, latent_dim)``.
        """
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample mean squared reconstruction error.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, input_dim)``.

        Returns
        -------
        torch.Tensor
            Per-sample MSE ``(B,)``.
        """
        x_hat = self(x)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

    def count_parameters(self) -> int:
        """Return total trainable parameters.

        Returns
        -------
        int
            Number of parameters with ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# TheftDetector — high-level wrapper
# ---------------------------------------------------------------------------

class TheftDetector(nn.Module):
    """Autoencoder-based anomaly detector compatible with the training pipeline.

    Trains exclusively on normal (non-theft) data.  At inference time, an
    elevated reconstruction error signals an anomalous (theft) event.

    Parameters
    ----------
    input_dim : int
        Number of features per sample.  If the input has shape
        ``(B, seq_len, input_dim)`` (sequence), it will be flattened to
        ``(B, seq_len * input_dim)`` automatically.
    seq_len : int
        Sequence length.  Used together with ``input_dim`` to determine the
        flat input size when sequence data is provided.
    hidden_dims : List[int], optional
        Encoder hidden + latent dims. Default ``[128, 64, 32]``.
    dropout : float
        Dropout probability. Default ``0.2``.
    use_batch_norm : bool
        Toggle BatchNorm. Default ``True``.
    lr : float
        Learning rate used in ``fit()``. Default ``1e-3``.
    weight_decay : float
        L2 regularisation in ``fit()``. Default ``1e-5``.

    Attributes
    ----------
    threshold_ : float
        Anomaly threshold (set by ``set_threshold()`` or a default percentile).
    min_error_ : float
        Minimum reconstruction error seen during fit (for normalisation).
    max_error_ : float
        Maximum reconstruction error seen during fit (for normalisation).
    is_fitted_ : bool
        Whether ``fit()`` has been called.

    Examples
    --------
    >>> detector = TheftDetector(input_dim=32, seq_len=10)
    >>> X_normal = torch.randn(200, 10, 32)
    >>> detector.fit(X_normal, epochs=5, batch_size=32)
    >>> proba = detector.predict_proba(torch.randn(16, 10, 32))
    >>> proba.shape
    torch.Size([16, 1])
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.flat_dim = input_dim * seq_len
        self.lr = lr
        self.weight_decay = weight_decay

        self.autoencoder = Autoencoder(
            input_dim=self.flat_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

        # Will be set during fit() / set_threshold()
        self.threshold_: float = float("inf")
        self.min_error_: float = 0.0
        self.max_error_: float = 1.0
        self.is_fitted_: bool = False

        logger.info(
            "TheftDetector initialised | input_dim=%d | seq_len=%d "
            "| flat_dim=%d | hidden_dims=%s | params=%d",
            input_dim,
            seq_len,
            self.flat_dim,
            hidden_dims,
            self.count_parameters(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten 3-D sequence tensor or validate 2-D input.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, seq_len, input_dim)`` or ``(B, flat_dim)``.

        Returns
        -------
        torch.Tensor
            ``(B, flat_dim)``.
        """
        if x.dim() == 3:
            return x.reshape(x.size(0), -1)
        if x.dim() == 2:
            if x.size(1) != self.flat_dim:
                raise ValueError(
                    f"Expected flat_dim={self.flat_dim}, got {x.size(1)}."
                )
            return x
        raise ValueError(f"Expected 2D or 3D tensor, got {x.dim()}D.")

    def _compute_errors(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample reconstruction errors (no grad).

        Parameters
        ----------
        x : torch.Tensor
            ``(B, ...)``

        Returns
        -------
        torch.Tensor
            ``(B,)`` MSE errors.
        """
        self.autoencoder.eval()
        with torch.no_grad():
            x_flat = self._flatten(x)
            errors = self.autoencoder.reconstruction_error(x_flat)
        return errors

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_normal: torch.Tensor,
        epochs: int = 50,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ) -> "TheftDetector":
        """Train the autoencoder on normal (non-theft) samples only.

        Parameters
        ----------
        X_normal : torch.Tensor
            Normal training data ``(N, seq_len, input_dim)`` or
            ``(N, flat_dim)``.
        epochs : int
            Number of training epochs. Default ``50``.
        batch_size : int
            Mini-batch size. Default ``64``.
        device : torch.device, optional
            Device for training.  Defaults to CUDA if available.
        verbose : bool
            Log epoch-level training loss. Default ``True``.

        Returns
        -------
        TheftDetector
            Self (for method chaining).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_flat = self._flatten(X_normal).to(device)
        self.autoencoder.to(device)

        dataset = TensorDataset(X_flat)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        optimiser = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=5, verbose=False
        )

        logger.info(
            "TheftDetector.fit() | device=%s | samples=%d | epochs=%d",
            device,
            len(X_flat),
            epochs,
        )

        for epoch in range(1, epochs + 1):
            self.autoencoder.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                optimiser.zero_grad()
                x_hat = self.autoencoder(batch)
                loss = F.mse_loss(x_hat, batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                optimiser.step()
                epoch_loss += loss.item() * batch.size(0)

            epoch_loss /= len(X_flat)
            scheduler.step(epoch_loss)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs):
                logger.info("  Epoch %3d/%d | loss=%.6f", epoch, epochs, epoch_loss)

        # Compute error stats on training data for normalisation
        errors = self._compute_errors(X_normal.to(device)).cpu().numpy()
        self.min_error_ = float(errors.min())
        self.max_error_ = float(np.percentile(errors, 99))  # clip at 99th percentile

        # Default threshold: 95th percentile of normal errors
        self.threshold_ = float(np.percentile(errors, 95))

        self.is_fitted_ = True
        logger.info(
            "TheftDetector.fit() complete | min_err=%.4f | max_err=%.4f "
            "| default_threshold=%.4f",
            self.min_error_,
            self.max_error_,
            self.threshold_,
        )
        return self

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Compute normalised anomaly probability for each sample.

        Reconstruction error is min-max normalised using statistics from
        ``fit()`` and clipped to ``[0, 1]``.

        Parameters
        ----------
        X : torch.Tensor
            Input ``(B, seq_len, input_dim)`` or ``(B, flat_dim)``.

        Returns
        -------
        torch.Tensor
            Anomaly probability ``(B, 1)`` with values in ``[0, 1]``.
            Higher → more likely theft.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "TheftDetector is not fitted. Call fit() before predict_proba()."
            )
        device = next(self.autoencoder.parameters()).device
        errors = self._compute_errors(X.to(device)).cpu()

        denom = max(self.max_error_ - self.min_error_, 1e-8)
        proba = (errors - self.min_error_) / denom
        proba = proba.clamp(0.0, 1.0)
        return proba.unsqueeze(-1)  # (B, 1)

    # ------------------------------------------------------------------
    # set_threshold
    # ------------------------------------------------------------------

    def set_threshold(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        percentile: float = 95.0,
    ) -> float:
        """Set the anomaly threshold using validation data.

        The threshold is chosen as the ``percentile``-th percentile of
        reconstruction errors on the *normal* validation samples
        (``y_val == 0``).  If no normal samples exist, falls back to
        the ``percentile``-th percentile of all errors.

        Parameters
        ----------
        X_val : torch.Tensor
            Validation input ``(N, seq_len, input_dim)`` or ``(N, flat_dim)``.
        y_val : torch.Tensor
            Ground-truth labels ``(N,)`` or ``(N, 1)`` — ``0`` = normal,
            ``1`` = theft.
        percentile : float
            Percentile of normal reconstruction errors to use as threshold.
            Default ``95.0``.

        Returns
        -------
        float
            The selected threshold value.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "TheftDetector is not fitted. Call fit() before set_threshold()."
            )

        device = next(self.autoencoder.parameters()).device
        errors = self._compute_errors(X_val.to(device)).cpu().numpy()

        labels = y_val.cpu().numpy().flatten()
        normal_mask = labels == 0
        if normal_mask.sum() > 0:
            normal_errors = errors[normal_mask]
            self.threshold_ = float(np.percentile(normal_errors, percentile))
        else:
            logger.warning(
                "No normal samples in y_val; using all errors for threshold."
            )
            self.threshold_ = float(np.percentile(errors, percentile))

        logger.info(
            "TheftDetector threshold set to %.4f (percentile=%.1f)",
            self.threshold_,
            percentile,
        )
        return self.threshold_

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return binary predictions: 1 = theft (anomaly), 0 = normal.

        Parameters
        ----------
        X : torch.Tensor
            Input ``(B, seq_len, input_dim)`` or ``(B, flat_dim)``.

        Returns
        -------
        torch.Tensor
            Binary labels ``(B, 1)`` as ``torch.long``.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "TheftDetector is not fitted. Call fit() before predict()."
            )
        device = next(self.autoencoder.parameters()).device
        errors = self._compute_errors(X.to(device)).cpu()

        # Normalised threshold for comparison against raw errors
        preds = (errors > self.threshold_).long().unsqueeze(-1)  # (B, 1)
        return preds

    # ------------------------------------------------------------------
    # forward (pipeline compatibility)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns ``predict_proba(x)`` for pipeline compat.

        Parameters
        ----------
        x : torch.Tensor
            Input ``(B, seq_len, input_dim)`` or ``(B, flat_dim)``.

        Returns
        -------
        torch.Tensor
            Anomaly probability ``(B, 1)``.
        """
        return self.predict_proba(x)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return total number of trainable parameters.

        Returns
        -------
        int
            Number of parameters with ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_latent_representations(self, X: torch.Tensor) -> torch.Tensor:
        """Encode samples to the latent space.

        Useful for visualisation (e.g. t-SNE / UMAP) of normal vs.
        anomalous representations.

        Parameters
        ----------
        X : torch.Tensor
            Input ``(B, seq_len, input_dim)`` or ``(B, flat_dim)``.

        Returns
        -------
        torch.Tensor
            Latent codes ``(B, latent_dim)``.
        """
        device = next(self.autoencoder.parameters()).device
        self.autoencoder.eval()
        with torch.no_grad():
            x_flat = self._flatten(X.to(device))
            z = self.autoencoder.encode(x_flat)
        return z.cpu()
