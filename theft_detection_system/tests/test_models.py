"""
test_models.py
==============
Pytest unit tests for all deep-learning model classes in the SGCC theft-
detection system.  Tests use small random PyTorch tensors only — no real data
required.

Models tested
-------------
  * CNN1D
  * LSTMClassifier
  * GRUClassifier
  * BiLSTMClassifier
  * CNNLSTMClassifier
  * CNNGRUClassifier
  * TransformerClassifier  (if present)
  * TCNClassifier          (if present)

Conventions
-----------
  - All inputs: ``(B=4, seq_len=32, input_dim=16)``
  - All expected outputs: ``(4, 1)``  with values in [0, 1]  (sigmoid)
  - ``count_parameters()`` must return a positive integer for every model

Run from any working directory:
    pytest theft_detection_system/tests/test_models.py -v
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Callable, Type

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_SYSTEM_ROOT = _THIS_DIR.parent
_REPO_ROOT = _SYSTEM_ROOT.parent

for _p in [str(_SYSTEM_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Eager imports for models that are always expected to exist
# ---------------------------------------------------------------------------
from src.models.deep_learning.cnn   import CNN1D              # noqa: E402
from src.models.deep_learning.lstm  import LSTMClassifier     # noqa: E402
from src.models.deep_learning.gru   import GRUClassifier      # noqa: E402
from src.models.deep_learning.bilstm  import BiLSTMClassifier  # noqa: E402
from src.models.deep_learning.cnn_lstm import CNNLSTMClassifier  # noqa: E402

# Optional models — may not be written yet
try:
    from src.models.deep_learning.cnn_gru import CNNGRUClassifier
    _HAS_CNN_GRU = True
except ImportError:
    _HAS_CNN_GRU = False

try:
    from src.models.deep_learning.transformer import TransformerClassifier
    _HAS_TRANSFORMER = True
except ImportError:
    _HAS_TRANSFORMER = False

try:
    from src.models.deep_learning.tcn import TCNClassifier
    _HAS_TCN = True
except ImportError:
    _HAS_TCN = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared test hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 4
SEQ_LEN    = 32
INPUT_DIM  = 16
DEVICE     = torch.device("cpu")


# ===========================================================================
# ── FIXTURES ────────────────────────────────────────────────────────────────
# ===========================================================================

@pytest.fixture(scope="module")
def dummy_input() -> torch.Tensor:
    """Standard random input tensor (B, seq_len, input_dim)."""
    torch.manual_seed(0)
    return torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)


# ---------------------------------------------------------------------------
# Individual model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cnn_model() -> CNN1D:
    return CNN1D(
        input_dim=INPUT_DIM,
        seq_len=SEQ_LEN,
        channels=[32, 64, 128],
        kernel_size=3,
        pool_size=2,
        dropout=0.0,       # disable dropout for deterministic tests
        fc_hidden=64,
        use_batch_norm=True,
    ).eval()


@pytest.fixture(scope="module")
def lstm_model() -> LSTMClassifier:
    return LSTMClassifier(
        input_dim=INPUT_DIM,
        seq_len=SEQ_LEN,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
        use_batch_norm=True,
        fc_hidden1=32,
        fc_hidden2=16,
    ).eval()


@pytest.fixture(scope="module")
def gru_model() -> GRUClassifier:
    return GRUClassifier(
        input_dim=INPUT_DIM,
        seq_len=SEQ_LEN,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
        use_batch_norm=True,
        fc_hidden1=32,
        fc_hidden2=16,
    ).eval()


@pytest.fixture(scope="module")
def bilstm_model() -> BiLSTMClassifier:
    return BiLSTMClassifier(
        input_dim=INPUT_DIM,
        seq_len=SEQ_LEN,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
        use_batch_norm=True,
        fc_hidden1=32,
        fc_hidden2=16,
    ).eval()


@pytest.fixture(scope="module")
def cnn_lstm_model() -> CNNLSTMClassifier:
    return CNNLSTMClassifier(
        input_dim=INPUT_DIM,
        seq_len=SEQ_LEN,
        cnn_channels=[32, 64],
        lstm_hidden=32,
        lstm_num_layers=2,
        dropout=0.0,
        use_batch_norm=True,
        fc_hidden=32,
    ).eval()


@pytest.fixture(scope="module")
def cnn_gru_model():
    if not _HAS_CNN_GRU:
        pytest.skip("CNNGRUClassifier not implemented yet")
    return CNNGRUClassifier(
        input_dim=INPUT_DIM,
        seq_len=SEQ_LEN,
        cnn_channels=[32, 64],
        gru_hidden=32,
        gru_layers=2,
        dropout=0.0,
        use_batch_norm=True,
        fc_hidden=32,
    ).eval()


@pytest.fixture(scope="module")
def transformer_model():
    if not _HAS_TRANSFORMER:
        pytest.skip("TransformerClassifier not implemented yet")
    return TransformerClassifier(
        input_dim=INPUT_DIM,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
    ).eval()


@pytest.fixture(scope="module")
def tcn_model():
    if not _HAS_TCN:
        pytest.skip("TCNClassifier not implemented yet")
    return TCNClassifier(
        input_dim=INPUT_DIM,
        num_channels=[32, 64, 128],
        kernel_size=3,
        dropout=0.0,
    ).eval()


# ===========================================================================
# ── HELPER ──────────────────────────────────────────────────────────────────
# ===========================================================================

def _forward_no_grad(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a forward pass without tracking gradients."""
    with torch.no_grad():
        return model(x.to(DEVICE))


def _assert_output_shape(out: torch.Tensor, expected_shape: tuple, model_name: str) -> None:
    assert out.shape == torch.Size(expected_shape), (
        f"{model_name}: expected output shape {expected_shape}, got {tuple(out.shape)}"
    )


def _assert_output_in_zero_one(out: torch.Tensor, model_name: str) -> None:
    assert out.min().item() >= 0.0 - 1e-6, (
        f"{model_name}: output min {out.min().item():.6f} < 0 (sigmoid broken?)"
    )
    assert out.max().item() <= 1.0 + 1e-6, (
        f"{model_name}: output max {out.max().item():.6f} > 1 (sigmoid broken?)"
    )


def _assert_param_count_positive(model: nn.Module, model_name: str) -> None:
    n = model.count_parameters()
    assert isinstance(n, int), (
        f"{model_name}.count_parameters() must return int, got {type(n)}"
    )
    assert n > 0, (
        f"{model_name}.count_parameters() returned {n}; expected > 0"
    )


# ===========================================================================
# ── CNN1D TESTS ─────────────────────────────────────────────────────────────
# ===========================================================================

class TestCNN1D:
    """Tests for the 1-D CNN binary classifier."""

    def test_forward_output_shape(
        self, cnn_model: CNN1D, dummy_input: torch.Tensor
    ) -> None:
        """Forward pass must produce shape (B, 1)."""
        out = _forward_no_grad(cnn_model, dummy_input)
        _assert_output_shape(out, (BATCH_SIZE, 1), "CNN1D")

    def test_output_in_zero_one(
        self, cnn_model: CNN1D, dummy_input: torch.Tensor
    ) -> None:
        """All outputs must be in [0, 1] (sigmoid activated)."""
        out = _forward_no_grad(cnn_model, dummy_input)
        _assert_output_in_zero_one(out, "CNN1D")

    def test_count_parameters_positive(self, cnn_model: CNN1D) -> None:
        """count_parameters() must return a positive integer."""
        _assert_param_count_positive(cnn_model, "CNN1D")

    def test_forward_no_nan(
        self, cnn_model: CNN1D, dummy_input: torch.Tensor
    ) -> None:
        """Output must not contain NaN or Inf values."""
        out = _forward_no_grad(cnn_model, dummy_input)
        assert torch.isfinite(out).all(), "CNN1D output contains NaN or Inf"

    def test_batch_size_1(self, cnn_model: CNN1D) -> None:
        """Model must handle batch size of 1 without errors."""
        x = torch.randn(1, SEQ_LEN, INPUT_DIM)
        out = _forward_no_grad(cnn_model, x)
        assert out.shape == torch.Size([1, 1]), (
            f"CNN1D: expected (1, 1) for batch_size=1, got {tuple(out.shape)}"
        )

    def test_gradient_flows(self, cnn_model: CNN1D, dummy_input: torch.Tensor) -> None:
        """Gradients must flow through the model (no dead sub-graphs)."""
        model_train = CNN1D(
            input_dim=INPUT_DIM, seq_len=SEQ_LEN,
            channels=[32, 64, 128], dropout=0.0, fc_hidden=64,
        ).train()
        x = dummy_input.clone().requires_grad_(True)
        out = model_train(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "CNN1D: no gradient flowed back to input"


# ===========================================================================
# ── LSTMClassifier TESTS ────────────────────────────────────────────────────
# ===========================================================================

class TestLSTMClassifier:
    """Tests for the LSTM binary classifier."""

    def test_forward_output_shape(
        self, lstm_model: LSTMClassifier, dummy_input: torch.Tensor
    ) -> None:
        out = _forward_no_grad(lstm_model, dummy_input)
        _assert_output_shape(out, (BATCH_SIZE, 1), "LSTMClassifier")

    def test_output_in_zero_one(
        self, lstm_model: LSTMClassifier, dummy_input: torch.Tensor
    ) -> None:
        out = _forward_no_grad(lstm_model, dummy_input)
        _assert_output_in_zero_one(out, "LSTMClassifier")

    def test_count_parameters_positive(self, lstm_model: LSTMClassifier) -> None:
        _assert_param_count_positive(lstm_model, "LSTMClassifier")

    def test_output_no_nan(
        self, lstm_model: LSTMClassifier, dummy_input: torch.Tensor
    ) -> None:
        out = _forward_no_grad(lstm_model, dummy_input)
        assert torch.isfinite(out).all(), "LSTMClassifier output contains NaN or Inf"

    def test_single_layer_no_dropout_warning(self) -> None:
        """Single-layer LSTM should not emit dropout-related warnings."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = LSTMClassifier(
                input_dim=INPUT_DIM, seq_len=SEQ_LEN,
                hidden_dim=32, num_layers=1, dropout=0.5,
            )
        # There should be no UserWarning from PyTorch about dropout on 1-layer LSTM
        dropout_warnings = [x for x in w if "dropout" in str(x.message).lower()]
        assert len(dropout_warnings) == 0, (
            "Unexpected dropout warning for single-layer LSTM"
        )

    def test_get_sequence_output_shape(
        self, lstm_model: LSTMClassifier, dummy_input: torch.Tensor
    ) -> None:
        """get_sequence_output() must return (lstm_out, pred) with correct shapes."""
        with torch.no_grad():
            lstm_out, pred = lstm_model.get_sequence_output(dummy_input)
        assert lstm_out.ndim == 3, (
            f"lstm_out must be 3-D (B, T, H), got shape {tuple(lstm_out.shape)}"
        )
        assert pred.shape == torch.Size([BATCH_SIZE, 1]), (
            f"pred shape mismatch: {tuple(pred.shape)}"
        )


# ===========================================================================
# ── GRUClassifier TESTS ─────────────────────────────────────────────────────
# ===========================================================================

class TestGRUClassifier:
    """Tests for the GRU binary classifier."""

    def test_forward_output_shape(
        self, gru_model: GRUClassifier, dummy_input: torch.Tensor
    ) -> None:
        out = _forward_no_grad(gru_model, dummy_input)
        _assert_output_shape(out, (BATCH_SIZE, 1), "GRUClassifier")

    def test_output_in_zero_one(
        self, gru_model: GRUClassifier, dummy_input: torch.Tensor
    ) -> None:
        out = _forward_no_grad(gru_model, dummy_input)
        _assert_output_in_zero_one(out, "GRUClassifier")

    def test_count_parameters_positive(self, gru_model: GRUClassifier) -> None:
        _assert_param_count_positive(gru_model, "GRUClassifier")

    def test_output_no_nan(
        self, gru_model: GRUClassifier, dummy_input: torch.Tensor
    ) -> None:
        out = _forward_no_grad(gru_model, dummy_input)
        assert torch.isfinite(out).all(), "GRUClassifier output contains NaN or Inf"

    def test_gru_fewer_params_than_lstm(self) -> None:
        """
        A GRU has fewer parameters than a same-size LSTM
        (GRU has 3 gates vs LSTM's 4).
        """
        gru  = GRUClassifier(input_dim=INPUT_DIM, seq_len=SEQ_LEN, hidden_dim=64)
        lstm = LSTMClassifier(input_dim=INPUT_DIM, seq_len=SEQ_LEN, hidden_dim=64)
        assert gru.count_parameters() < lstm.count_parameters(), (
            "GRU should have fewer parameters than equivalent LSTM"
        )

    def test_get_sequence_output_shape(
        self, gru_model: GRUClassifier, dummy_input: torch.Tensor
    ) -> None:
        with torch.no_grad():
            gru_out, pred = gru_model.get_sequence_output(dummy_input)
        assert gru_out.ndim == 3, (
            f"gru_out must be 3-D, got {tuple(gru_out.shape)}"
        )
        assert pred.shape == torch.Size([BATCH_SIZE, 1])


# ===========================================================================
# ── BiLSTMClassifier TESTS ──────────────────────────────────────────────────
# ===========================================================================

class TestBiLSTMClassifier:
    """Tests for the Bidirectional LSTM binary classifier."""

    def test_forward_output_shape(
        self, bilstm_model: BiLSTMClassifier, dummy_input: torch.Tensor
    ) -> None:
        out = _forward_no_grad(bilstm_model, dummy_input)
        _assert_output_shape(out, (BATCH_SIZE, 1), "BiLSTMClassifier")

    def test_output_in_zero_one(
        self, bilstm_model: BiLSTMClassifier, dummy_input: torch.Tensor
    ) -> None:
        out = _forward_no_grad(bilstm_model, dummy_input)
        _assert_output_in_zero_one(out, "BiLSTMClassifier")

    def test_count_parameters_positive(self, bilstm_model: BiLSTMClassifier) -> None:
        _assert_param_count_positive(bilstm_model, "BiLSTMClassifier")

    def test_output_no_nan(
        self, bilstm_model: BiLSTMClassifier, dummy_input: torch.Tensor
    ) -> None:
        out = _forward_no_grad(bilstm_model, dummy_input)
        assert torch.isfinite(out).all(), "BiLSTMClassifier output contains NaN or Inf"

    def test_more_params_than_unidirectional(self) -> None:
        """
        BiLSTM must have more parameters than a same-size unidirectional LSTM
        because it has both forward and backward cells.
        """
        bi   = BiLSTMClassifier(input_dim=INPUT_DIM, seq_len=SEQ_LEN, hidden_dim=32)
        uni  = LSTMClassifier(input_dim=INPUT_DIM,   seq_len=SEQ_LEN, hidden_dim=32)
        assert bi.count_parameters() > uni.count_parameters(), (
            "BiLSTM should have more parameters than unidirectional LSTM"
        )

    def test_get_sequence_output_bilateral_shape(
        self, bilstm_model: BiLSTMClassifier, dummy_input: torch.Tensor
    ) -> None:
        """BiLSTM sequence output must have last dim == hidden_dim * 2."""
        with torch.no_grad():
            bilstm_out, pred = bilstm_model.get_sequence_output(dummy_input)
        expected_hidden = bilstm_model.hidden_dim * 2
        assert bilstm_out.shape[-1] == expected_hidden, (
            f"BiLSTM out channels: expected {expected_hidden}, got {bilstm_out.shape[-1]}"
        )


# ===========================================================================
# ── CNNLSTMClassifier TESTS ─────────────────────────────────────────────────
# ===========================================================================

class TestCNNLSTMClassifier:
    """Tests for the CNN-LSTM hybrid binary classifier."""

    def test_forward_output_shape(
        self, cnn_lstm_model: CNNLSTMClassifier, dummy_input: torch.Tensor
    ) -> None:
        out = _forward_no_grad(cnn_lstm_model, dummy_input)
        _assert_output_shape(out, (BATCH_SIZE, 1), "CNNLSTMClassifier")

    def test_output_in_zero_one(
        self, cnn_lstm_model: CNNLSTMClassifier, dummy_input: torch.Tensor
    ) -> None:
        out = _forward_no_grad(cnn_lstm_model, dummy_input)
        _assert_output_in_zero_one(out, "CNNLSTMClassifier")

    def test_count_parameters_positive(self, cnn_lstm_model: CNNLSTMClassifier) -> None:
        _assert_param_count_positive(cnn_lstm_model, "CNNLSTMClassifier")

    def test_output_no_nan(
        self, cnn_lstm_model: CNNLSTMClassifier, dummy_input: torch.Tensor
    ) -> None:
        out = _forward_no_grad(cnn_lstm_model, dummy_input)
        assert torch.isfinite(out).all(), "CNNLSTMClassifier output contains NaN or Inf"

    def test_get_cnn_features_shape(
        self, cnn_lstm_model: CNNLSTMClassifier, dummy_input: torch.Tensor
    ) -> None:
        """get_cnn_features() must return a 3-D tensor (B, T', C)."""
        with torch.no_grad():
            cnn_feats = cnn_lstm_model.get_cnn_features(dummy_input)
        assert cnn_feats.ndim == 3, (
            f"CNN features must be 3-D (B, T', C), got {tuple(cnn_feats.shape)}"
        )
        assert cnn_feats.shape[0] == BATCH_SIZE, (
            f"Batch dimension mismatch in CNN features: {cnn_feats.shape[0]}"
        )


# ===========================================================================
# ── CNNGRUClassifier TESTS ──────────────────────────────────────────────────
# ===========================================================================

class TestCNNGRUClassifier:
    """Tests for the CNN-GRU hybrid binary classifier."""

    def test_forward_output_shape(self, cnn_gru_model, dummy_input) -> None:
        out = _forward_no_grad(cnn_gru_model, dummy_input)
        _assert_output_shape(out, (BATCH_SIZE, 1), "CNNGRUClassifier")

    def test_output_in_zero_one(self, cnn_gru_model, dummy_input) -> None:
        out = _forward_no_grad(cnn_gru_model, dummy_input)
        _assert_output_in_zero_one(out, "CNNGRUClassifier")

    def test_count_parameters_positive(self, cnn_gru_model) -> None:
        _assert_param_count_positive(cnn_gru_model, "CNNGRUClassifier")

    def test_output_no_nan(self, cnn_gru_model, dummy_input) -> None:
        out = _forward_no_grad(cnn_gru_model, dummy_input)
        assert torch.isfinite(out).all(), "CNNGRUClassifier output contains NaN or Inf"


# ===========================================================================
# ── TransformerClassifier TESTS ─────────────────────────────────────────────
# ===========================================================================

class TestTransformerClassifier:
    """Tests for the Transformer binary classifier."""

    def test_forward_output_shape(self, transformer_model, dummy_input) -> None:
        out = _forward_no_grad(transformer_model, dummy_input)
        _assert_output_shape(out, (BATCH_SIZE, 1), "TransformerClassifier")

    def test_output_in_zero_one(self, transformer_model, dummy_input) -> None:
        """Transformer output must be in [0, 1] (sigmoid applied)."""
        out = _forward_no_grad(transformer_model, dummy_input)
        _assert_output_in_zero_one(out, "TransformerClassifier")

    def test_count_parameters_positive(self, transformer_model) -> None:
        _assert_param_count_positive(transformer_model, "TransformerClassifier")

    def test_output_no_nan(self, transformer_model, dummy_input) -> None:
        out = _forward_no_grad(transformer_model, dummy_input)
        assert torch.isfinite(out).all(), (
            "TransformerClassifier output contains NaN or Inf"
        )

    def test_output_is_probability(self, transformer_model, dummy_input) -> None:
        """Every scalar in the output must represent a valid probability in [0, 1]."""
        out = _forward_no_grad(transformer_model, dummy_input)
        probs = out.squeeze(-1).numpy()
        for i, p in enumerate(probs):
            assert 0.0 <= p <= 1.0, (
                f"TransformerClassifier: sample {i} output {p:.6f} not in [0, 1]"
            )

    def test_batch_determinism(self, transformer_model, dummy_input) -> None:
        """Identical inputs in eval mode must produce identical outputs."""
        out1 = _forward_no_grad(transformer_model, dummy_input)
        out2 = _forward_no_grad(transformer_model, dummy_input)
        assert torch.allclose(out1, out2), (
            "TransformerClassifier: non-deterministic output in eval mode"
        )


# ===========================================================================
# ── TCNClassifier TESTS ─────────────────────────────────────────────────────
# ===========================================================================

class TestTCNClassifier:
    """Tests for the Temporal Convolutional Network binary classifier."""

    def test_forward_output_shape(self, tcn_model, dummy_input) -> None:
        out = _forward_no_grad(tcn_model, dummy_input)
        _assert_output_shape(out, (BATCH_SIZE, 1), "TCNClassifier")

    def test_output_in_zero_one(self, tcn_model, dummy_input) -> None:
        out = _forward_no_grad(tcn_model, dummy_input)
        _assert_output_in_zero_one(out, "TCNClassifier")

    def test_count_parameters_positive(self, tcn_model) -> None:
        _assert_param_count_positive(tcn_model, "TCNClassifier")

    def test_output_no_nan(self, tcn_model, dummy_input) -> None:
        out = _forward_no_grad(tcn_model, dummy_input)
        assert torch.isfinite(out).all(), "TCNClassifier output contains NaN or Inf"


# ===========================================================================
# ── PARAMETRIC TESTS (run each model through a common suite) ────────────────
# ===========================================================================

@pytest.mark.parametrize("model_fixture_name", [
    "cnn_model",
    "lstm_model",
    "gru_model",
    "bilstm_model",
    "cnn_lstm_model",
])
class TestCommonModelBehaviours:
    """
    Common behavioural assertions that every model must satisfy.
    Parameterised over the always-present model fixtures.
    """

    def test_output_shape_parametric(
        self, request, model_fixture_name: str, dummy_input: torch.Tensor
    ) -> None:
        """All models must output shape (B, 1)."""
        model = request.getfixturevalue(model_fixture_name)
        out = _forward_no_grad(model, dummy_input)
        assert out.shape == torch.Size([BATCH_SIZE, 1]), (
            f"{model_fixture_name}: expected output shape ({BATCH_SIZE}, 1), "
            f"got {tuple(out.shape)}"
        )

    def test_output_range_parametric(
        self, request, model_fixture_name: str, dummy_input: torch.Tensor
    ) -> None:
        """All models must output values in [0, 1]."""
        model = request.getfixturevalue(model_fixture_name)
        out = _forward_no_grad(model, dummy_input)
        _assert_output_in_zero_one(out, model_fixture_name)

    def test_param_count_positive_parametric(
        self, request, model_fixture_name: str
    ) -> None:
        """count_parameters() must return positive integer for all models."""
        model = request.getfixturevalue(model_fixture_name)
        _assert_param_count_positive(model, model_fixture_name)

    def test_eval_mode_no_nan(
        self, request, model_fixture_name: str, dummy_input: torch.Tensor
    ) -> None:
        """No NaN / Inf in output when model is in eval mode."""
        model = request.getfixturevalue(model_fixture_name)
        out = _forward_no_grad(model, dummy_input)
        assert torch.isfinite(out).all(), (
            f"{model_fixture_name}: output contains NaN or Inf in eval mode"
        )

    def test_is_subclass_of_nn_module(
        self, request, model_fixture_name: str
    ) -> None:
        """Every model class must be a subclass of nn.Module."""
        model = request.getfixturevalue(model_fixture_name)
        assert isinstance(model, nn.Module), (
            f"{model_fixture_name} must be an nn.Module subclass"
        )


# ===========================================================================
# ── ENTRY POINT ─────────────────────────────────────────────────────────────
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
