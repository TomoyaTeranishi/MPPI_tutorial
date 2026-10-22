from pathlib import Path

import pytest
import torch

from scripts.models.pendulum_dynamics_mlp import PendulumDynamicsMLP
from scripts.models.pendulum_neural_dynamics import PendulumNeuralDynamics
from scripts.models.pendulum_normalization import PendulumNormalization


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

NORMALIZATION_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "datasets"
    / "pendulum"
    / "random_hold1_v001"
    / "splits"
    / "normalization.npz"
)


def test_mlp_output_shape() -> None:
    """MLPが入力数に対応する3次元出力を返すことを確認する。"""

    model = PendulumDynamicsMLP()
    model_input = torch.zeros(
        size=(8, 4),
        dtype=torch.float32,
    )

    model_output = model(model_input)

    assert model_output.shape == (8, 3)
    assert torch.isfinite(model_output).all()


def test_normalization_round_trip() -> None:
    """標準化と逆標準化で元の入力へ戻ることを確認する。"""

    normalization = PendulumNormalization.from_npz(
        NORMALIZATION_PATH
    )

    physical_input = torch.tensor(
        [
            [0.0, -1.0, 0.5, 1.0],
            [0.2, -0.9, -0.5, -1.0],
        ],
        dtype=torch.float32,
    )

    standardized_input = normalization.standardize_input(
        physical_input
    )

    restored_input = normalization.inverse_standardize_input(
        standardized_input
    )

    assert torch.allclose(
        restored_input,
        physical_input,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_single_state_prediction() -> None:
    """単一状態から有限な次状態を返すことを確認する。"""

    torch.manual_seed(0)

    model = PendulumNeuralDynamics(
        normalization_path=NORMALIZATION_PATH
    )

    next_state = model(
        theta=torch.tensor(torch.pi),
        theta_dot=torch.tensor(0.0),
        torque=torch.tensor(1.0),
    )

    assert next_state.shape == (2,)
    assert torch.isfinite(next_state).all()
    assert -torch.pi <= next_state[0] <= torch.pi
    assert -8.0 <= next_state[1] <= 8.0


def test_batch_prediction() -> None:
    """複数状態を一括推論できることを確認する。"""

    model = PendulumNeuralDynamics(
        normalization_path=NORMALIZATION_PATH
    )

    batch_size = 1000

    next_states = model(
        theta=torch.full(
            size=(batch_size,),
            fill_value=torch.pi,
        ),
        theta_dot=torch.zeros(batch_size),
        torque=torch.linspace(
            start=-2.0,
            end=2.0,
            steps=batch_size,
        ),
    )

    assert next_states.shape == (batch_size, 2)
    assert torch.isfinite(next_states).all()
    assert torch.all(next_states[:, 0] >= -torch.pi)
    assert torch.all(next_states[:, 0] <= torch.pi)
    assert torch.all(next_states[:, 1] >= -8.0)
    assert torch.all(next_states[:, 1] <= 8.0)


def test_encoded_output_is_on_unit_circle() -> None:
    """予測したsin・cosが単位円上にあることを確認する。"""

    model = PendulumNeuralDynamics(
        normalization_path=NORMALIZATION_PATH
    )

    encoded_output = model.predict_encoded_next_state(
        theta=torch.tensor([torch.pi, 2.0, 0.0]),
        theta_dot=torch.tensor([0.0, 1.0, -1.0]),
        torque=torch.tensor([1.0, -1.0, 0.5]),
    )

    squared_norm = (
        encoded_output[:, 0].square()
        + encoded_output[:, 1].square()
    )

    assert torch.allclose(
        squared_norm,
        torch.ones_like(squared_norm),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_gradient_reaches_all_mlp_parameters() -> None:
    """損失からMLPの全パラメータへ勾配が流れることを確認する。"""

    model = PendulumNeuralDynamics(
        normalization_path=NORMALIZATION_PATH
    )

    prediction = model(
        theta=torch.tensor([torch.pi, 2.0]),
        theta_dot=torch.tensor([0.0, 1.0]),
        torque=torch.tensor([1.0, -1.0]),
    )

    loss = prediction.square().mean()
    loss.backward()

    parameters = list(model.model.parameters())

    assert len(parameters) == 6
    assert all(
        parameter.grad is not None
        for parameter in parameters
    )
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in parameters
    )


def test_invalid_mlp_input_shape() -> None:
    """入力次元が4でない場合にエラーとなることを確認する。"""

    model = PendulumDynamicsMLP()

    with pytest.raises(ValueError):
        model(
            torch.zeros(
                size=(8, 3),
                dtype=torch.float32,
            )
        )