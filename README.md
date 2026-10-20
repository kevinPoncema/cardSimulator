# Simulador de Carreras 2D con Telemetría LIDAR (Fase 1)

Núcleo jugable de un simulador de carreras 2D con vista cenital (*top-down*) en Python, diseñado con **Arquitectura Orientada al Dominio (DDD)**, principios **SOLID** estrictos, físicas **arcade simplificadas y divertidas** sobre **Pymunk (Chipmunk 2D)**, cámara de seguimiento sobre **mapas de gran escala** con **Pygame-CE** y un sistema de telemetría sensorial **LIDAR 2D** listo para modelos de Aprendizaje por Refuerzo (RL).

---

## 1. Estructura del Proyecto y Documentación

Toda la documentación técnica detallada se encuentra en la carpeta [`docs/`](docs):

* **[`docs/architecture.md`](docs/architecture.md):** **Arquitectura integral del sistema**, capas DDD/Clean Architecture, SOLID, puertos y adaptadores, ciclo de simulación y diagramas Mermaid.
* **[`docs/system_design.md`](docs/system_design.md):** Especificación de contratos de interfaces, principios SOLID y secuencias.
* **[`docs/physics_model.md`](docs/physics_model.md):** Modelo físico arcade asistido en Pymunk, ruedas virtuales, fricción lateral y superficies.
* **[`docs/rendering_and_maps.md`](docs/rendering_and_maps.md):** Sistema de renderizado, cámara con interpolación suave, minimapa y formato de los 12 circuitos.
* **[`docs/lidar_spec.md`](docs/lidar_spec.md):** Especificación matemática de los 5 rayos LIDAR y vector de estado $S \in \mathbb{R}^7$.
* **[`docs/rl_training_guide.md`](docs/rl_training_guide.md):** Guía de entrenamiento por refuerzo (RL), red neuronal MLP y tabla de recompensas.
* **[`docs/training_philosophy.md`](docs/training_philosophy.md):** **Filosofía del entrenamiento**, psicología del diseño de recompensas, prevención de trampas (*anti-reward hacking*), caza de monedas y auto-recuperación de trompos.

```text
├── assets/
│   ├── maps/                     # 12 circuitos variados en formato JSON (pequeños, medianos y grandes)
│   └── sprites/                  # Sprites generados para la paleta de vehículos
├── config/
│   └── reward_config.json        # Pesos de recompensas y castigos fácilmente editables sin tocar código
├── docs/                         # Documentación técnica completa
├── models/
│   ├── rl_agent_v1.npz           # Pesos de la Red Neuronal entrenada (Versión 1)
│   └── training_history.json     # Histórico de rendimiento del entrenamiento RL
├── src/
│   ├── application/              # Orquestador de carrera, telemetría y subsistema RL
│   ├── domain/                   # Lógica pura del negocio (DDD, sin Pygame ni Pymunk)
│   ├── infrastructure/           # Adaptadores técnicos (Pymunk, Pygame-CE, LIDAR, Red Neuronal)
│   ├── train_agent.py            # CLI de entrenamiento multinúcleo de la Red Neuronal
│   └── main.py                   # Raíz de composición (DIP) y punto de entrada del juego
├── tests/                        # 34 Pruebas unitarias automatizadas con pytest
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 2. Instalación y Puesta a Punto

```bash
# 1. Crear entorno virtual local
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias exactas
pip install -r requirements.txt

# 3. Verificar que los 34 tests unitarios pasen correctamente
.venv/bin/pytest tests/ -v
```

---

## 3. Comandos de Ejecución de la Simulación (`src/main.py`)

El punto de entrada [`src/main.py`](src/main.py) permite configurar completamente la experiencia de juego o evaluación mediante parámetros de línea de comandos:

### Tabla de Parámetros de `src/main.py`

| Parámetro | Tipo / Opciones | Por Defecto | Descripción |
| :--- | :--- | :---: | :--- |
| `--map` | Nombre del circuito | `kart_classic` | Circuito a cargar (12 disponibles en catálogo). |
| `--num-cars` | Entero $\ge 1$ | `1` | **Número de coches simultáneos en pista**. Se colocan en una parrilla de salida escalonada. |
| `--controller` | `human`, `rl_ai`, `dummy_ai` | `human` | Controlador del vehículo principal. Los coches adicionales se controlan automáticamente por la red neuronal (`rl_ai`). |
| `--vs-ai` | Flag booleano | `False` | Añade un rival IA azul en la parrilla (equivalente rápido a `--num-cars 2`). |
| `--color` | Nombre o `#HEX` | `crimson_red` | Color del vehículo principal (`crimson_red`, `electric_blue`, `emerald_green`, `amber_gold`, `royal_purple` o `#RRGGBB`). |
| `--model-path` | Ruta a `.npz` | `models/rl_agent_v1.npz` | Archivo con los pesos de la red neuronal a utilizar para los coches IA. |
| `--lidar-range` | Flotante | `300.0` | Alcance máximo en píxeles de los 5 rayos sensoriales LIDAR. |
| `--max-frames` | Entero | `0` (ilimitado) | Fotogramas máximos a simular antes de cerrar (útil para benchmarks o pruebas automatizadas). |
| `--headless` | Flag booleano | `False` | Ejecuta la simulación en segundo plano sin abrir ventana gráfica. |

---

### Ejemplos Prácticos de Comandos de Simulación

#### 1. Conducir como Humano en Solitario
```bash
# Ejecutar en el circuito clásico
.venv/bin/python3 src/main.py

# Ejecutar en el circuito de curvas cerradas (Mapa 3) con coche color esmeralda
.venv/bin/python3 src/main.py --map pocket_chicane --color emerald_green
```

#### 2. Carrera Humano contra Varios Coches IA y Recolección de Monedas
```bash
# Parrilla de 4 coches en el circuito Pocket Chicane
.venv/bin/python3 src/main.py --controller human --map pocket_chicane --num-cars 4
```
*Tú conduces el coche rojo principal (`car_player_1`), y los otros 3 coches rivales son controlados por la Red Neuronal entrenada ([`models/rl_agent_v1.npz`](models/rl_agent_v1.npz)).*

#### 3. Ver a la Red Neuronal Conducir Sola (Demostración Autónoma)
```bash
# 1 coche IA conduciendo de forma autónoma
.venv/bin/python3 src/main.py --controller rl_ai --map kart_classic

# 6 coches IA compitiendo entre sí en el MegaRing Internacional
.venv/bin/python3 src/main.py --controller rl_ai --map megaring_international --num-cars 6
```

#### 4. Duelo Rápido 1 vs 1 contra la IA
```bash
.venv/bin/python3 src/main.py --vs-ai --map retro_grand_prix
```

---

## 4. Controles y Elementos en Pantalla

### Controles del Teclado
| Tecla | Acción |
| :--- | :--- |
| `W` / `Flecha Arriba` | Acelerar hacia adelante |
| `S` / `Flecha Abajo` | Frenar / Marcha atrás |
| `A` / `Flecha Izquierda` | Girar a la izquierda |
| `D` / `Flecha Derecha` | Girar a la derecha |
| `Espacio` | **Freno de mano / Modo Derrape asistido** |
| `M` / `N` | **Siguiente mapa / Mapa anterior** en tiempo real |
| `1` a `9`, `0`, `-`, `=` | **Acceso directo a cualquiera de los 12 mapas** |
| `R` | **Reiniciar carrera** (restablece posiciones, monedas y obstáculos) |
| `F3` | **Alternar HUD de Depuración Visual:** Rayos LIDAR y vector $S$ |
| `ESC` | Salir del simulador |

### Panel de Estado en Vivo (Esquina Inferior Derecha)
En la esquina inferior derecha se muestra un panel flotante con la telemetría en tiempo real de cada coche en pista:
* **`#` y Color:** Identificador del vehículo en la parrilla.
* **`Vuelta`:** Número de vuelta actual (valida que se crucen todos los checkpoints en orden antes de contar vuelta completa).
* **`CP`:** Checkpoints superados en la vuelta actual (`X/3`).
* **`Monedas`:** Cantidad de monedas doradas recolectadas.
* **`Pts`:** Puntuación acumulada (suma puntos por vueltas válidas y por cada moneda recogida).

### Monedas Doradas (`Coin`)
En todos los circuitos aparecen monedas doradas que se generan **estrictamente sobre asfalto alcanzable** (nunca fuera de pista ni sobre obstáculos). Al tocarlas con el vehículo, suman puntos tanto a la puntuación del juego como a la función de recompensa del entrenamiento RL, y se reubican automáticamente en un nuevo punto libre de la pista.

---

## 5. Catálogo de los 12 Circuitos Disponibles

El simulador cuenta con 12 circuitos en [`assets/maps/`](assets/maps), organizados por escala y accesibles mediante teclas rápidas o el parámetro `--map`:

| Tecla | Identificador | Nombre del Circuito | Dimensiones | Características y Obstáculos |
| :---: | :--- | :--- | :---: | :--- |
| `1` | `kart_classic` | Mini Karting Classic | $1600 \times 1200\text{ px}$ | Pista compacta ideal para pruebas rápidas y warm-up. |
| `2` | `oval_arena` | Thunder Oval Arena | $1600 \times 1100\text{ px}$ | Óvalo de alta velocidad con chicanes interiores y conos. |
| `3` | `pocket_chicane` | Pocket Chicane Circuit | $1800 \times 1300\text{ px}$ | Curvas técnicas cerradas con baches y barreras de hormigón. |
| `4` | `retro_grand_prix` | Retro GP Classic | $2600 \times 1800\text{ px}$ | Trazado clásico de gran escala con rectas largas y horquillas. |
| `5` | `neon_speedway` | Neon Speedway | $2800 \times 1800\text{ px}$ | Pista rápida de fluidez continua con zonas de aceite. |
| `6` | `canyon_drift` | Canyon Drift Circuit | $2800 \times 2000\text{ px}$ | Circuito sinuoso optimizado para derrapes y control lateral. |
| `7` | `starlight_loop` | Starlight Grand Loop | $2600 \times 2000\text{ px}$ | Doble lazo con chicanes alternadas y conos dinámicos. |
| `8` | `desert_dunes` | Desert Dunes Ring | $2800 \times 1900\text{ px}$ | Asfalto ancho con curvas abiertas y pianos pronunciados. |
| `9` | `dockyard_sprint` | Dockyard Industrial | $2700 \times 1900\text{ px}$ | Zona portuaria con curvas de $90^\circ$ y barreras rígidas. |
| `0` | `megaring_international` | MegaRing International GP | $3600 \times 2500\text{ px}$ | Supercircuito estilo Fórmula 1 de más de $3.6\text{ km}$ virtuales. |
| `-` | `volcano_ridge` | Volcano Ridge Expressway | $3800 \times 2600\text{ px}$ | Autopista gigante perimetral con desniveles visuales y chicanes. |
| `=` | `monaco_night` | Monaco Night Grand Prix | $3500 \times 2500\text{ px}$ | Circuito urbano monumental con múltiples variantes y obstáculos. |

---

## 6. Comandos de Entrenamiento por Refuerzo (`src/train_agent.py`)

El script [`src/train_agent.py`](src/train_agent.py) ejecuta el entrenamiento evolutivo / CEM-NES de la red neuronal aprovechando todos los núcleos de tu CPU mediante `multiprocessing`.

### Tabla de Parámetros de `src/train_agent.py`

| Parámetro | Tipo | Por Defecto | Descripción |
| :--- | :---: | :---: | :--- |
| `--maps` | Lista de strings | `kart_classic oval_arena pocket_chicane` | Circuitos en los que se evaluará cada candidato de la población. |
| `--generations` | Entero | `12` | Número de generaciones evolutivas del entrenamiento. |
| `--population` | Entero | `20` | Tamaño de la población de redes neuronales evaluadas por generación. |
| `--max-steps` | Entero | `750` | Pasos máximos de simulación física por episodio en cada circuito. |
| `--workers` | Entero | Núcleos CPU detectados | Número de procesos paralelos de evaluación simultánea en CPU. |
| `--output` | Ruta archivo | `models/rl_agent_v1.npz` | Ruta donde se guardan los mejores pesos sinápticos obtenidos. |
| `--rewards-config` | Ruta archivo | `config/reward_config.json` | Archivo JSON con los pesos de recompensas y castigos. |

---

### Ejemplos de Comandos de Entrenamiento

#### 1. Reentrenamiento Rápido con los Circuitos Pequeños (Recomendado)
```bash
.venv/bin/python3 src/train_agent.py \
  --maps kart_classic pocket_chicane oval_arena \
  --generations 12 \
  --population 16 \
  --max-steps 700 \
  --workers 8
```
*Tarda menos de un minuto en un procesador Ryzen 7 y actualiza [`models/rl_agent_v1.npz`](models/rl_agent_v1.npz) con el mejor modelo capaz de esquivar obstáculos, recoger monedas y desatascarse con reversa en las curvas cerradas del Mapa 3.*

#### 2. Entrenamiento Extensivo en Circuitos Medianos y Grandes
```bash
.venv/bin/python3 src/train_agent.py \
  --maps kart_classic retro_grand_prix canyon_drift megaring_international \
  --generations 30 \
  --population 32 \
  --max-steps 1200 \
  --workers 8
```

#### 3. Ajuste de Recompensas en `config/reward_config.json`
Puedes modificar directamente los números en [`config/reward_config.json`](config/reward_config.json):
* `checkpoint_reward` (`+65.0`): Premio por cruzar un checkpoint en orden.
* `lap_completion_reward` (`+250.0`): Premio por completar una vuelta válida.
* `coin_collected_reward` (`+180.0`): **Premio masivo por recoger una moneda transitable** (incentiva activamente a la IA a desviarse para cazarlas).
* `coin_proximity_weight` (`+2.5`): **Guía continua de atracción** hacia monedas visibles por delante del vehículo.
* `reverse_finish_line_penalty` (`-1500.0`): **Castigo colosal y aborto del episodio** por cruzar la línea de meta en sentido contrario.
* `wrong_way_timeout_penalty` (`-1000.0`): Castigo por persistir en sentido contrario.
* `wall_stuck_step_penalty` (`-14.0`): Castigo progresivo por quedarse atascado contra un muro sin retroceder.
* `reverse_escape_reward` (`+4.5`): Premio por dar marcha atrás para separarse de un obstáculo frontal.
* `cone_collision_penalty` (`-40.0`) y `barrier_collision_penalty` (`-75.0`): Castigos por chocar con obstáculos.

---

## 7. Ejecución de Pruebas Unitarias (`pytest`)

El proyecto cuenta con una suite completa de **35 tests unitarios** que verifican la arquitectura DDD, objetos de valor, cinemática, cálculo LIDAR, entorno de RL, monedas, atascos y recuperación de trompos:

```bash
# Ejecutar toda la batería de tests (35 passed)
.venv/bin/pytest tests/ -v

# Ejecutar únicamente los tests del dominio vehicular y pureza DDD
.venv/bin/pytest tests/test_domain_car.py -v

# Ejecutar los tests de geometría y normalización LIDAR
.venv/bin/pytest tests/test_lidar_calculations.py -v

# Ejecutar los tests del sistema de RL, recompensas y monedas
.venv/bin/pytest tests/test_rl_training.py -v
```
