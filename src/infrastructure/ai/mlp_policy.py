"""Red Neuronal Perceptrón Multicapa (MLP) para política de conducción continua RL."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.domain.value_objects.car_action import CarAction


class MLPPolicy:
    """Red Neuronal MLP (7 -> H1 -> H2 -> 2) que mapea el vector LIDAR S a CarAction.

    Entradas (7):
        S = [d_0, d_45_left, d_45_right, d_90_left, d_90_right, v_norm, omega_norm]
    Salidas (2, vía tanh en [-1.0, 1.0]):
        out[0]: Acelerador (>0) / Freno (<0)
        out[1]: Dirección (steer en [-1.0, 1.0])
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dims: tuple[int, ...] = (32, 32),
        output_dim: int = 2,
        seed: int | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.output_dim = int(output_dim)

        rng = np.random.default_rng(seed)
        layer_dims = [self.input_dim, *self.hidden_dims, self.output_dim]

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for i in range(len(layer_dims) - 1):
            in_d, out_d = layer_dims[i], layer_dims[i + 1]
            scale = np.sqrt(2.0 / (in_d + out_d))
            w = rng.normal(0.0, scale, size=(in_d, out_d)).astype(np.float32)
            b = np.zeros(out_d, dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

    @classmethod
    def create_warm_start(
        cls,
        hidden_dims: tuple[int, ...] = (32, 32),
        seed: int = 42,
    ) -> MLPPolicy:
        """Crea una red neuronal inicializada con simetría sensorial para acelerar el arranque RL."""
        policy = cls(input_dim=7, hidden_dims=hidden_dims, output_dim=2, seed=seed)
        # Escalar pesos aleatorios iniciales a una magnitud suave y configurar neuronas guía
        for i in range(len(policy.weights)):
            policy.weights[i] *= 0.15

        # Neurona 0: detector de vía libre frontal (d_0 - velocidad excesiva) -> aceleración
        policy.weights[0][0, 0] = 1.8    # d_0
        policy.weights[0][1, 0] = 0.5    # d_45_left
        policy.weights[0][2, 0] = 0.5    # d_45_right
        policy.weights[0][5, 0] = -0.9   # v_norm (frena si va muy rápido ante curva cerrada)
        policy.biases[0][0] = 0.25

        # Neurona 1: detector de curva / esquiva diagonal (d_45_left - d_45_right)
        policy.weights[0][1, 1] = 2.2    # d_45_left (+45°)
        policy.weights[0][2, 1] = -2.2   # d_45_right (-45°)
        policy.weights[0][3, 1] = 0.9    # d_90_left (+90°)
        policy.weights[0][4, 1] = -0.9   # d_90_right (-90°)
        policy.weights[0][6, 1] = -0.35  # amortiguación de velocidad angular

        # Neurona 2: detector de emergencia frontal (obstáculo inminente en d_0)
        policy.weights[0][0, 2] = -2.5   # se activa fuerte cuando d_0 es pequeño
        policy.biases[0][2] = 0.9

        if len(policy.weights) == 2:
            policy.weights[1][0, 0] = 1.5
            policy.weights[1][1, 1] = 1.6
            policy.weights[1][2, 0] = -0.8
        else:
            # Propagar canales guía a través de la segunda capa oculta
            policy.weights[1][0, 0] = 1.4
            policy.weights[1][1, 1] = 1.4
            policy.weights[1][2, 2] = 1.4

            policy.weights[2][0, 0] = 1.5
            policy.weights[2][1, 1] = 1.6
            policy.weights[2][2, 0] = -0.8
            policy.biases[2][0] = 0.35

        return policy

    @property
    def num_parameters(self) -> int:
        """Retorna el número total de pesos y sesgos entrenables de la red."""
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases, strict=True))

    def get_flat_params(self) -> np.ndarray:
        """Empaqueta todos los pesos y sesgos en un único vector 1D float32."""
        parts: list[np.ndarray] = []
        for w, b in zip(self.weights, self.biases, strict=True):
            parts.append(w.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts).astype(np.float32)

    def set_flat_params(self, flat_params: np.ndarray) -> None:
        """Restaura los pesos y sesgos de la red a partir de un vector 1D."""
        arr = np.asarray(flat_params, dtype=np.float32).ravel()
        if arr.size != self.num_parameters:
            raise ValueError(
                f"Tamaño de parámetros inválido: esperado {self.num_parameters}, recibido {arr.size}."
            )
        idx = 0
        for i in range(len(self.weights)):
            w_shape = self.weights[i].shape
            w_size = self.weights[i].size
            self.weights[i] = arr[idx : idx + w_size].reshape(w_shape).copy()
            idx += w_size

            b_shape = self.biases[i].shape
            b_size = self.biases[i].size
            self.biases[i] = arr[idx : idx + b_size].reshape(b_shape).copy()
            idx += b_size

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Ejecuta el paso hacia adelante (forward pass) y retorna acción continua en [-1, 1]^2."""
        x = np.asarray(obs, dtype=np.float32).reshape(-1, self.input_dim)
        for w, b in zip(self.weights, self.biases, strict=True):
            x = np.tanh(x @ w + b)
        return x[0] if obs.ndim == 1 else x

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        """Interfaz compatible con Stable-Baselines3: retorna (action, state)."""
        action = self.forward(obs)
        return action, None

    def predict_car_action(self, obs: np.ndarray) -> CarAction:
        """Convierte directamente la observación S en un CarAction ejecutable."""
        raw = self.forward(obs)
        longitudinal = float(np.clip(raw[0], -1.0, 1.0))
        steer = float(np.clip(raw[1], -1.0, 1.0))

        # Si el frente está muy cerca de un obstáculo y el giro es pronunciado, activar freno de mano suave
        d_0 = float(obs[0]) if obs.size > 0 else 1.0
        v_norm = float(obs[5]) if obs.size > 5 else 0.0
        handbrake = bool(d_0 < 0.32 and v_norm > 0.45 and abs(steer) > 0.72)

        throttle = max(0.0, longitudinal)
        brake = max(0.0, -longitudinal)
        return CarAction(
            throttle=throttle,
            brake=brake,
            steer=steer,
            handbrake=handbrake,
        )

    def save(self, path: str | Path) -> Path:
        """Guarda los pesos y arquitectura de la red neuronal en formato comprimido .npz."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict: dict[str, np.ndarray] = {
            "input_dim": np.array(self.input_dim, dtype=np.int32),
            "hidden_dims": np.array(self.hidden_dims, dtype=np.int32),
            "output_dim": np.array(self.output_dim, dtype=np.int32),
            "flat_params": self.get_flat_params(),
        }
        np.savez(out_path, **save_dict)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> MLPPolicy:
        """Carga una red neuronal previamente entrenada desde un archivo .npz."""
        file_path = Path(path)
        data = np.load(file_path)
        input_dim = int(data["input_dim"])
        hidden_dims = tuple(int(x) for x in data["hidden_dims"])
        output_dim = int(data["output_dim"])

        policy = cls(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
        policy.set_flat_params(data["flat_params"])
        return policy
