"""Script de entrenamiento local de la Red Neuronal (Versión 1) mediante Aprendizaje por Refuerzo."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.rl.reward_config import RewardConfig
from src.application.rl.trainer import RLTrainer
from src.domain.entities.track import PRESET_MAPS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento local de la Red Neuronal RL para conducción y esquiva de obstáculos."
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        choices=list(PRESET_MAPS.keys()),
        default=["kart_classic", "pocket_chicane"],
        help="Circuitos sobre los cuales entrenar la red neuronal.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=12,
        help="Número de generaciones de entrenamiento (por defecto: 12).",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=20,
        help="Tamaño de la población de redes evaluadas por generación (por defecto: 20).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=850,
        help="Pasos físicos máximos por episodio de evaluación (por defecto: 850).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Número de procesos paralelos en CPU (ideal para Ryzen 7).",
    )
    parser.add_argument(
        "--reward-config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "reward_config.json"),
        help="Ruta al archivo JSON con los pesos de recompensas y castigos.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "models" / "rl_agent_v1.npz"),
        help="Ruta donde se guardará la red neuronal entrenada (.npz).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg_path = Path(args.reward_config)
    if cfg_path.exists():
        reward_cfg = RewardConfig.from_json_file(cfg_path)
    else:
        reward_cfg = RewardConfig()
        reward_cfg.to_json_file(cfg_path)

    print("=====================================================================")
    print(" ENTRENAMIENTO DE RED NEURONAL RL (VERSIÓN 1) - SIMULADOR 2D LIDAR")
    print("=====================================================================")
    print(f" * Circuitos de entrenamiento : {', '.join(args.maps)}")
    print(f" * Generaciones x Población   : {args.generations} x {args.population}")
    print(f" * Procesos paralelos (CPU)   : {args.workers}")
    print(f" * Archivo de Recompensas     : {cfg_path}")
    print(f" * Modelo de salida           : {args.output}")
    print("---------------------------------------------------------------------")

    trainer = RLTrainer(
        map_keys=args.maps,
        reward_config=reward_cfg,
        hidden_dims=(32, 32),
        population_size=args.population,
        max_steps_per_episode=args.max_steps,
        num_workers=args.workers,
    )

    trainer.train(
        generations=args.generations,
        output_model_path=args.output,
        history_path=PROJECT_ROOT / "models" / "training_history.json",
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
