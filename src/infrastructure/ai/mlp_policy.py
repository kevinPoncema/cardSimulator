"""Red Neuronal Perceptrón Multicapa (MLP) con maniobra de reversa y desatasco para RL."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.domain.value_objects.car_action import CarAction


class MLPPolicy:
    """Red Neuronal MLP (7 -> H1 -> H2 -> 2) que mapea el vector LIDAR S a CarAction.

    Incluye memoria de desatasco para aprender y ejecutar maniobras de reversa + giro
    invertido cuando el frente del vehículo queda bloqueado contra una pared en curvas
    cerradas (evitando quedarse atascado haciendo micro-correcciones).
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

        # Estado interno de maniobra de reversa y desatasco
        self._reverse_escape_steps: int = 0
        self._stalled_steps: int = 0
        self._escape_steer_dir: float = 1.0

    def reset_state(self) -> None:
        """Reinicia la memoria temporal de desatasco al comenzar un nuevo episodio."""
        self._reverse_escape_steps = 0
        self._stalled_steps = 0
        self._escape_steer_dir = 1.0

    @classmethod
    def create_warm_start(
        cls,
        hidden_dims: tuple[int, ...] = (32, 32),
        seed: int = 42,
    ) -> MLPPolicy:
        """Crea una red neuronal inicializada con simetría sensorial y frenado en curvas cerradas."""
        policy = cls(input_dim=7, hidden_dims=hidden_dims, output_dim=2, seed=seed)
        for i in range(len(policy.weights)):
            policy.weights[i] *= 0.12

        # Neurona 0: Control longitudinal (acelera en recta despejada, modula velocidad en curva)
        policy.weights[0][0, 0] = 2.0    # d_0 frontal
        policy.weights[0][1, 0] = 0.45   # d_45_left
        policy.weights[0][2, 0] = 0.45   # d_45_right
        policy.weights[0][5, 0] = -1.05  # v_norm (evita exceso de velocidad al entrar a horquillas)
        policy.biases[0][0] = 0.20

        # Neurona 1: Giro anticipado en curvas y esquiva diagonal (d_45_left - d_45_right)
        policy.weights[0][1, 1] = 2.6    # d_45_left (+45°)
        policy.weights[0][2, 1] = -2.6   # d_45_right (-45°)
        policy.weights[0][3, 1] = 1.25   # d_90_left (+90°)
        policy.weights[0][4, 1] = -1.25  # d_90_right (-90°)
        policy.weights[0][6, 1] = -0.30  # amortiguación de velocidad angular

        # Neurona 2: Detector de pared frontal cercana (frenado / reversa si d_0 es bajo)
        policy.weights[0][0, 2] = -3.2   # se activa fuerte cuando d_0 < 0.28
        policy.biases[0][2] = 0.85

        # Neurona 3: Giro de emergencia lateral cuando el frente se cierra
        policy.weights[0][3, 3] = 1.9    # d_90_left
        policy.weights[0][4, 3] = -1.9   # d_90_right

        if len(policy.weights) == 2:
            policy.weights[1][0, 0] = 1.5
            policy.weights[1][1, 1] = 1.8
            policy.weights[1][2, 0] = -1.4
            policy.weights[1][3, 1] = 0.8
        else:
            policy.weights[1][0, 0] = 1.5
            policy.weights[1][1, 1] = 1.6
            policy.weights[1][2, 2] = 1.6
            policy.weights[1][3, 3] = 1.4

            policy.weights[2][0, 0] = 1.5
            policy.weights[2][1, 1] = 1.8
            policy.weights[2][2, 0] = -1.35
            policy.weights[2][3, 1] = 0.75
            policy.biases[2][0] = 0.28

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
        """Convierte la observación S en un CarAction con maniobra inteligente de reversa y corrección."""
        raw = self.forward(obs)
        longitudinal = float(np.clip(raw[0], -1.0, 1.0))
        steer = float(np.clip(raw[1], -1.0, 1.0))

        if obs.size >= 7:
            d_0 = float(obs[0])
            d_45_l = float(obs[1])
            d_45_r = float(obs[2])
            d_90_l = float(obs[3])
            d_90_r = float(obs[4])
            v_norm = float(obs[5])

            # Determinar hacia qué lado hay mayor espacio abierto para orientar el morro
            open_side_signal = (d_45_l - d_45_r) * 1.4 + (d_90_l - d_90_r) * 1.1
            preferred_dir = 1.0 if open_side_signal >= 0.0 else -1.0

            # Detectar si el coche se ha chocado o estancado frente a un muro/obstáculo
            if d_0 < 0.16 or (d_0 < 0.28 and abs(v_norm) < 0.05):
                self._stalled_steps += 1
            else:
                self._stalled_steps = max(0, self._stalled_steps - 1)

            # Activar maniobra de reversa + desatasco si está bloqueado frente al muro
            if self._reverse_escape_steps == 0 and (d_0 < 0.14 or self._stalled_steps >= 6):
                self._reverse_escape_steps = 24
                self._escape_steer_dir = preferred_dir
                self._stalled_steps = 0

            if self._reverse_escape_steps > 0:
                self._reverse_escape_steps -= 1
                # Si ya abrió suficiente espacio frontal, terminar reversa antes y avanzar corrigiendo
                if d_0 > 0.38 and self._reverse_escape_steps < 14:
                    self._reverse_escape_steps = 0
                else:
                    # En marcha atrás (v_fwd < 0), el giro en Pymunk invierte su signo (direction_sign = -1).
                    # Por tanto, invertimos el volante (-_escape_steer_dir) mientras retrocede para que
                    # la trompa del coche gire apuntando hacia el lado despejado de la pista.
                    return CarAction(
                        throttle=0.0,
                        brake=0.85,
                        steer=-self._escape_steer_dir,
                        handbrake=False,
                    )

            # Si acaba de salir o el frente aún es reducido (0.14 <= d_0 < 0.35), avanzar poco a poco corrigiendo
            if d_0 < 0.34:
                cautious_throttle = max(0.32, min(0.55, longitudinal))
                strong_steer = np.clip(steer * 0.5 + open_side_signal * 1.3, -1.0, 1.0)
                return CarAction(
                    throttle=float(cautious_throttle),
                    brake=0.0,
                    steer=float(strong_steer),
                    handbrake=bool(d_0 < 0.26 and v_norm > 0.38 and abs(strong_steer) > 0.75),
                )

        handbrake = bool(
            obs.size >= 6 and float(obs[0]) < 0.32 and float(obs[5]) > 0.48 and abs(steer) > 0.75
        )
        throttle = max(0.0, longitudinal)
        brake = max(0.0, -longitudinal)
        return CarAction(
            throttle=throttle,
            brake=brake,
            steer=steer,
            handbrake=handbrake,
        )

    def turnaround_action(self, obs: np.ndarray) -> CarAction:
        """Maniobra refleja de giro en U para reorientar el vehículo cuando queda encarado al revés."""
        d_45_l = float(obs[1]) if obs.size > 1 else 0.5
        d_45_r = float(obs[2]) if obs.size > 2 else 0.5
        turn_dir = 1.0 if d_45_l >= d_45_r else -1.0
        return CarAction(
            throttle=0.45,
            brake=0.0,
            steer=turn_dir,
            handbrake=False,
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
