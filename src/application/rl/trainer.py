"""Entrenador de Aprendizaje por Refuerzo (CEM-NES) para la Red Neuronal MLPPolicy."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.application.rl.car_racing_env import CarRacingRLEnv
from src.application.rl.reward_config import RewardConfig
from src.domain.entities.track import Track
from src.infrastructure.ai.mlp_policy import MLPPolicy
from src.infrastructure.physics.pymunk_physics_engine import PymunkPhysicsEngine
from src.infrastructure.sensors.lidar_sensor_2d import LidarSensor2D


@dataclass
class GenerationMetrics:
    """Métricas registradas al finalizar cada generación de entrenamiento."""

    generation: int
    mean_reward: float
    best_gen_reward: float
    all_time_best_reward: float
    best_laps: int
    best_checkpoints: int
    elapsed_sec: float


def _evaluate_candidate_worker(args: tuple[np.ndarray, tuple[int, ...], list[str], dict[str, Any], int]) -> tuple[float, int, int]:
    """Función worker para evaluar un vector de pesos de red neuronal en episodios completos."""
    flat_params, hidden_dims, map_keys, reward_cfg_dict, max_steps = args

    policy = MLPPolicy(input_dim=7, hidden_dims=hidden_dims, output_dim=2)
    policy.set_flat_params(flat_params)

    reward_cfg = RewardConfig(**reward_cfg_dict)
    tracks = [Track.create_default_circuit(k) for k in map_keys]
    engine = PymunkPhysicsEngine()
    sensor = LidarSensor2D(engine, max_range=300.0)

    env = CarRacingRLEnv(
        physics_engine=engine,
        sensor=sensor,
        tracks=tracks,
        reward_config=reward_cfg,
        max_steps_per_episode=max_steps,
    )

    total_reward = 0.0
    total_laps = 0
    total_checkpoints = 0

    for t_idx in range(len(tracks)):
        policy.reset_state()
        obs, _ = env.reset(options={"track_index": t_idx})
        done = False
        while not done:
            action = policy.predict_car_action(obs)
            obs, r, terminated, truncated, info = env.step(action)
            total_reward += r
            if terminated or truncated:
                done = True
                total_laps += int(info.get("laps_completed", info.get("score", 0)))
                total_checkpoints += int(info.get("total_checkpoints", 0))

    n_tracks = max(1, len(tracks))
    return (total_reward / n_tracks, total_laps, total_checkpoints)


class RLTrainer:
    """Entrena una red neuronal MLPPolicy optimizando la función de recompensa parametrizable."""

    def __init__(
        self,
        map_keys: list[str] | None = None,
        reward_config: RewardConfig | None = None,
        hidden_dims: tuple[int, ...] = (32, 32),
        population_size: int = 24,
        elite_fraction: float = 0.25,
        initial_sigma: float = 0.18,
        max_steps_per_episode: int = 900,
        num_workers: int = 1,
        seed: int = 42,
    ) -> None:
        self.map_keys = map_keys or ["kart_classic"]
        self.reward_config = reward_config or RewardConfig()
        self.hidden_dims = hidden_dims
        self.population_size = max(4, int(population_size))
        self.elite_count = max(2, int(round(self.population_size * elite_fraction)))
        self.sigma = float(initial_sigma)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.num_workers = max(1, int(num_workers))
        self.rng = np.random.default_rng(seed)

        self.base_policy = MLPPolicy.create_warm_start(hidden_dims=hidden_dims, seed=seed)
        self.mean_params = self.base_policy.get_flat_params()
        self.best_params = self.mean_params.copy()
        self.best_reward: float = -float("inf")
        self.history: list[GenerationMetrics] = []

    def train(
        self,
        generations: int = 15,
        output_model_path: str | Path = "models/rl_agent_v1.npz",
        history_path: str | Path = "models/training_history.json",
        verbose: bool = True,
    ) -> MLPPolicy:
        """Ejecuta el bucle de entrenamiento por generaciones y guarda el mejor modelo."""
        reward_cfg_dict = asdict(self.reward_config)
        start_time = time.perf_counter()

        for gen in range(1, generations + 1):
            gen_t0 = time.perf_counter()

            # El candidato 0 siempre incluye el mejor conjunto de pesos actual (elitismo estricto)
            candidates: list[np.ndarray] = [self.best_params.copy()]
            for _ in range(self.population_size - 1):
                noise = self.rng.normal(0.0, self.sigma, size=self.mean_params.shape).astype(
                    np.float32
                )
                candidates.append(self.mean_params + noise)

            worker_args = [
                (cand, self.hidden_dims, self.map_keys, reward_cfg_dict, self.max_steps_per_episode)
                for cand in candidates
            ]

            if self.num_workers > 1:
                with mp.Pool(processes=self.num_workers) as pool:
                    results = pool.map(_evaluate_candidate_worker, worker_args)
            else:
                results = [_evaluate_candidate_worker(arg) for arg in worker_args]

            rewards = np.array([res[0] for res in results], dtype=np.float32)
            laps = np.array([res[1] for res in results], dtype=np.int32)
            checkpoints = np.array([res[2] for res in results], dtype=np.int32)

            # Seleccionar los mejores candidatos (élite)
            elite_indices = np.argsort(rewards)[::-1][: self.elite_count]
            best_idx = int(elite_indices[0])
            gen_best_reward = float(rewards[best_idx])
            gen_mean_reward = float(np.mean(rewards))

            if gen_best_reward > self.best_reward:
                self.best_reward = gen_best_reward
                self.best_params = candidates[best_idx].copy()

            # Actualizar media ponderada por rango hacia los individuos élite
            weights = np.linspace(1.0, 0.3, num=self.elite_count, dtype=np.float32)
            weights /= np.sum(weights)
            new_mean = np.zeros_like(self.mean_params)
            for w_rank, idx in zip(weights, elite_indices, strict=True):
                new_mean += w_rank * candidates[int(idx)]

            self.mean_params = 0.3 * self.mean_params + 0.7 * new_mean
            self.sigma = max(0.04, self.sigma * 0.96)

            gen_elapsed = time.perf_counter() - gen_t0
            metrics = GenerationMetrics(
                generation=gen,
                mean_reward=round(gen_mean_reward, 2),
                best_gen_reward=round(gen_best_reward, 2),
                all_time_best_reward=round(self.best_reward, 2),
                best_laps=int(laps[best_idx]),
                best_checkpoints=int(checkpoints[best_idx]),
                elapsed_sec=round(gen_elapsed, 2),
            )
            self.history.append(metrics)

            if verbose:
                print(
                    f"[Gen {gen:02d}/{generations:02d}] "
                    f"Media: {metrics.mean_reward:+8.2f} | "
                    f"Mejor Gen: {metrics.best_gen_reward:+8.2f} | "
                    f"Récord: {metrics.all_time_best_reward:+8.2f} | "
                    f"Checkpoints: {metrics.best_checkpoints:2d} | "
                    f"Vueltas: {metrics.best_laps:1d} | "
                    f"Tiempo: {metrics.elapsed_sec:4.2f}s"
                )

        # Guardar el mejor modelo entrenado y su historial
        self.base_policy.set_flat_params(self.best_params)
        saved_model_path = self.base_policy.save(output_model_path)

        hist_path = Path(history_path)
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        hist_data = {
            "total_training_sec": round(time.perf_counter() - start_time, 2),
            "map_keys": self.map_keys,
            "best_reward": round(self.best_reward, 2),
            "reward_config": reward_cfg_dict,
            "generations": [asdict(m) for m in self.history],
        }
        hist_path.write_text(json.dumps(hist_data, indent=2), encoding="utf-8")

        if verbose:
            print(f"\nModelo entrenado guardado en: {saved_model_path}")
            print(f"Historial de métricas guardado en: {hist_path}")

        return self.base_policy
