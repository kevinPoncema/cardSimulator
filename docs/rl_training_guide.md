# Guía de Entrenamiento por Refuerzo (RL) y Configuración de Recompensas

## 1. Visión General del Sistema de Entrenamiento (Versión 1)

El simulador incorpora un sistema completo de **Aprendizaje por Refuerzo (*Reinforcement Learning*)** desacoplado en tres módulos principales:
1. **Configuración y Calculadora de Recompensas ([`RewardConfig` / `RewardCalculator`](file:///home/kevin/Documentos/personal/pythonZZZ/simulador%20carritos%20fuuui/src/application/rl/reward_config.py)):** Parametrizable mediante el archivo externo [`config/reward_config.json`](file:///home/kevin/Documentos/personal/pythonZZZ/simulador%20carritos%20fuuui/config/reward_config.json) sin necesidad de tocar el código fuente.
2. **Entorno de Simulación RL ([`CarRacingRLEnv`](file:///home/kevin/Documentos/personal/pythonZZZ/simulador%20carritos%20fuuui/src/application/rl/car_racing_env.py)):** Sigue el estándar de `Gymnasium` (`reset()`, `step(action)`, `render()`, `close()`), ejecutándose sin interfaz gráfica a miles de pasos por segundo.
3. **Red Neuronal MLP y Entrenador Multinúcleo ([`MLPPolicy`](file:///home/kevin/Documentos/personal/pythonZZZ/simulador%20carritos%20fuuui/src/infrastructure/ai/mlp_policy.py) y [`RLTrainer`](file:///home/kevin/Documentos/personal/pythonZZZ/simulador%20carritos%20fuuui/src/application/rl/trainer.py)):** Perceptrón Multicapa ($7 \to 32 \to 32 \to 2$) entrenado en paralelo en todos los núcleos del procesador mediante Estrategias Evolutivas y Método de Entropía Cruzada (*CEM-NES*).

```mermaid
flowchart LR
    JSON["config/reward_config.json"] --> RC["RewardCalculator"]
    ENV["CarRacingRLEnv (Pymunk + LIDAR)"] --> OBS["Vector S (7 dim)"]
    OBS --> MLP["Red Neuronal MLP (7 -> 32 -> 32 -> 2)"]
    MLP --> ACT["CarAction (Throttle, Brake, Steer)"]
    ACT --> ENV
    ENV --> RC
    RC --> TRAINER["RLTrainer (Multiprocessing CPU)"]
    TRAINER -->|Actualiza Pesos| MLP
    TRAINER -->|Guarda Modelo| NPZ["models/rl_agent_v1.npz"]
```

---

## 2. Arquitectura de la Red Neuronal (`MLPPolicy`)

- **Capa de Entrada ($7$ neuronas):** Recibe el vector de observación normalizado $S$:
  $$S = \left[ \frac{d_0}{R_{\max}}, \frac{d_{45^\circ}}{R_{\max}}, \frac{d_{-45^\circ}}{R_{\max}}, \frac{d_{90^\circ}}{R_{\max}}, \frac{d_{-90^\circ}}{R_{\max}}, \frac{v}{v_{\max}}, \frac{\omega}{\omega_{\max}} \right]$$
  > [!IMPORTANT]
  > Tanto los obstáculos sólidos (**muros, conos y barreras**) como los peligros de terreno (**baches gigantes y manchas de aceite**, registrados como sensores espaciales en Pymunk) son detectados por los 5 rayos `segment_query` del LIDAR. Esto permite que la red neuronal "vea" un bache o charco de aceite a distancia y gire para esquivarlo antes de pisarlo.
- **Capas Ocultas ($32 \times 32$ neuronas con activación $\tanh$):** Procesan la geometría de las curvas y la asimetría entre rayos izquierdos y derechos para decidir cuándo frenar antes de una horquilla y hacia qué lado esquivar un obstáculo.
- **Capa de Salida ($2$ neuronas en $[-1.0, 1.0]$):**
  - `out[0]`: Acelerador (si $>0$) o Freno/Reversa (si $<0$).
  - `out[1]`: Giro del volante (`steer` $\in [-1.0, 1.0]$).

---

## 3. Cómo Ajustar las Recompensas y Penalizaciones (`config/reward_config.json`)

Todos los premios y castigos se leen desde [`config/reward_config.json`](file:///home/kevin/Documentos/personal/pythonZZZ/simulador%20carritos%20fuuui/config/reward_config.json). Puedes editar cualquier número con un editor de texto y volver a lanzar `src/train_agent.py`:

| Parámetro en `reward_config.json` | Valor por Defecto | Qué hace exactamente |
| :--- | :---: | :--- |
| `forward_progress_weight` | `+1.6` | Premia avanzar a buena velocidad proyectada sobre el sentido de la pista. |
| `curve_alignment_weight` | `+0.7` | Premia mantener el chasis alineado con la tangente de cada curva. |
| `asphalt_center_bonus` | `+0.35` | Bonificación por mantenerse dentro del asfalto limpio. |
| `obstacle_clearance_bonus` | `+0.45` | Premia mantener despejados los rayos frontales/diagonales (esquivar obstáculos con margen). |
| `checkpoint_reward` | **`+65.0`** | **Premio grande al cruzar cada Checkpoint (`1 -> 2 -> 3`) en el sentido correcto.** |
| `lap_completion_reward` | **`+250.0`** | **Gran premio al completar una vuelta entera válida (cruzar Meta tras los 3 checkpoints).** |
| `cone_collision_penalty` | **`-40.0`** | **Quita puntos al chocar contra un cono de tráfico.** |
| `barrier_collision_penalty` | **`-75.0`** | **Quita puntos al estrellarse contra una barrera rígida.** |
| `wall_collision_penalty` | **`-65.0`** | **Quita puntos al chocar contra el muro perimetral de la pista.** |
| `pothole_step_penalty` | **`-4.0`** | **Quita puntos en cada frame que el coche pisa un bache gigante.** |
| `oil_slick_step_penalty` | **`-4.5`** | **Quita puntos en cada frame que el coche pisa una mancha de aceite.** |
| `grass_step_penalty` | `-1.5` | Quita puntos por salirse al césped. |
| `wrong_way_step_penalty` | **`-3.5`** | **Quita puntos en cada paso que el coche conduce en sentido contrario.** |
| `max_wrong_way_steps` | `90` | Número máximo de pasos seguidos permitidos en sentido contrario ($1.5\text{ s}$). |
| `wrong_way_timeout_penalty` | **`-90.0`** | **Castigo severo y fin del episodio si va en sentido contrario durante demasiado tiempo.** |
| `reverse_finish_line_penalty` | **`-120.0`** | **Castigo severo si cruza la línea de meta en sentido contrario.** |
| `reverse_checkpoint_penalty` | **`-55.0`** | **Castigo si cruza un checkpoint en sentido contrario.** |
| `terminate_on_wall_crash` | `true` | Si es `true`, el episodio termina inmediatamente al chocar contra el muro. |
| `terminate_on_barrier_crash` | `true` | Si es `true`, el episodio termina al chocar contra una barrera de hormigón. |
| `terminate_on_reverse_finish_line` | `true` | Si es `true`, el episodio termina si cruza la meta al revés. |

---

## 4. Comandos para Entrenar y Probar la IA en tu Máquina

### Paso 1: Ejecutar el entrenamiento en tu PC (usando los núcleos de tu Ryzen 7)
```bash
# Entrenamiento rápido inicial (guarda en models/rl_agent_v1.npz)
.venv/bin/python3 src/train_agent.py --maps kart_classic pocket_chicane --generations 12 --population 20 --workers 8
```

Si quieres hacer un entrenamiento más largo sobre circuitos medianos y grandes:
```bash
.venv/bin/python3 src/train_agent.py \
  --maps kart_classic retro_grand_prix canyon_drift megaring_international \
  --generations 30 \
  --population 32 \
  --max-steps 1200 \
  --workers 8
```

### Paso 2: Ver a tu Red Neuronal Conducir Sola
```bash
.venv/bin/python3 src/main.py --controller rl_ai --map kart_classic
```
*(Mientras la IA conduce, puedes pulsar `1-9, 0, -, =` o `M`/`N` para cambiarla de circuito en caliente y ver cómo reacciona a curvas y obstáculos nuevos, o `F3` para ver sus rayos LIDAR).*

### Paso 3: Competir Tú (Humano) contra tu Red Neuronal (`--vs-ai`)
```bash
.venv/bin/python3 src/main.py --vs-ai --map retro_grand_prix
```
Esto coloca tu coche rojo junto al coche azul controlado por `models/rl_agent_v1.npz` en la línea de salida.
