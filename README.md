# Simulador de Carreras 2D con Telemetría LIDAR (Fase 1)

Núcleo jugable de un simulador de carreras 2D con vista cenital (*top-down*) en Python, diseñado con **Arquitectura Orientada al Dominio (DDD)**, principios **SOLID** estrictos, físicas **arcade simplificadas y divertidas** sobre **Pymunk (Chipmunk 2D)**, cámara de seguimiento sobre **mapas de gran escala** con **Pygame-CE** y un sistema de telemetría sensorial **LIDAR 2D** listo para modelos de Aprendizaje por Refuerzo (RL).

---

## 1. Estructura del Proyecto y Documentación

```text
├── docs/
│   ├── system_design.md          # Diagramas de clases (Mermaid), contratos SOLID y secuencia
│   ├── physics_model.md          # Modelo físico arcade asistido, derrape y superficies
│   ├── rendering_and_maps.md     # Cómo funciona el renderizado, la cámara y la creación de mapas
│   └── lidar_spec.md             # Especificación matemática del LIDAR y vector de estado S
├── src/
│   ├── domain/                   # Lógica pura del negocio (sin dependencias de Pygame/Pymunk)
│   │   ├── entities/             # Car, CarConfig, Track, SensorReading, SensorRay
│   │   ├── value_objects/        # Vector2D, Speed, Angle, Color, CarAction
│   │   └── interfaces/           # IPhysicsEngine, ISensor, IRenderer, IInputController, ITelemetryProvider, IControllable
│   ├── infrastructure/           # Adaptadores técnicos
│   │   ├── physics/              # PymunkPhysicsEngine, PymunkCarBody
│   │   ├── sensors/              # LidarSensor2D (usando segment_query de Pymunk)
│   │   ├── rendering/            # PygameRenderer (cámara + minimapa), SpriteManager
│   │   └── input/                # HumanKeyboardController, DummyAiController
│   ├── application/              # Casos de uso y orquestación del bucle principal
│   │   ├── race_orchestrator.py
│   │   └── telemetry_collector.py
│   └── main.py                   # Punto de entrada / Raíz de composición (Composition Root)
├── assets/
│   ├── sprites/                  # Sprites PNG generados para la paleta de colores de vehículos
│   └── maps/                     # 4 circuitos de gran tamaño en formato JSON
├── tests/
│   ├── test_domain_car.py        # Tests de dominio, objetos de valor, física y pureza DIP
│   └── test_lidar_calculations.py # Tests de transformaciones local-a-global y normalización S
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 2. Instalación y Ejecución

```bash
# 1. Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias fijadas
pip install -r requirements.txt

# 3. Iniciar el simulador a 60 FPS
.venv/bin/python3 src/main.py
```

### Selección de Mapas y Catálogo de 12 Circuitos
El simulador incluye **12 circuitos variados organizados en 3 categorías** con obstáculos (conos dinámicos, barreras, baches gigantes y manchas de aceite):

- **3 Pequeños:**
  - `kart_classic` (`1`): **Mini Karting Classic** ($1600 \times 1200\text{ px}$)
  - `oval_arena` (`2`): **Thunder Oval Arena** ($1600 \times 1100\text{ px}$)
  - `pocket_chicane` (`3`): **Pocket Chicane Circuit** ($1800 \times 1300\text{ px}$)
- **6 Medianos:**
  - `retro_grand_prix` (`4`): **Retro GP Classic** ($2600 \times 1800\text{ px}$)
  - `neon_speedway` (`5`): **Neon Speedway** ($2800 \times 1800\text{ px}$)
  - `canyon_drift` (`6`): **Canyon Drift Circuit** ($2800 \times 2000\text{ px}$)
  - `starlight_loop` (`7`): **Starlight Grand Loop** ($2600 \times 2000\text{ px}$)
  - `desert_dunes` (`8`): **Desert Dunes Ring** ($2800 \times 1900\text{ px}$)
  - `dockyard_sprint` (`9`): **Dockyard Industrial** ($2700 \times 1900\text{ px}$)
- **3 Grandes:**
  - `megaring_international` (`0`): **MegaRing International GP** ($3600 \times 2500\text{ px}$)
  - `volcano_ridge` (`-`): **Volcano Ridge Expressway** ($3800 \times 2600\text{ px}$)
  - `monaco_night` (`=`): **Monaco Night Grand Prix** ($3500 \times 2500\text{ px}$)

*(Puedes cambiar de mapa en cualquier momento con las teclas directas `1-9, 0, -, =` o con `M` / `N`).*

```bash
# Iniciar directamente en un mapa y color específicos
.venv/bin/python3 src/main.py --map kart_classic --color electric_blue

# Iniciar en un circuito gigante GP
.venv/bin/python3 src/main.py --map megaring_international
```

### Controles en Carrera
| Tecla | Acción |
| :--- | :--- |
| `W` / `Flecha Arriba` | Acelerar |
| `S` / `Flecha Abajo` | Frenar / Reversa |
| `A` / `Flecha Izquierda` | Girar a la izquierda |
| `D` / `Flecha Derecha` | Girar a la derecha |
| `Espacio` | **Freno de mano / Modo Derrape** |
| `M` / `N` | **Mapa siguiente / Mapa anterior** |
| `1` a `9`, `0`, `-`, `=` | **Selección directa de los 12 mapas** |
| `R` | **Reiniciar posición del coche y obstáculos** |
| `F3` | Alternar el **Modo de Depuración Visual LIDAR** y **HUD del Vector $S$** |
| `ESC` | Salir del simulador |

---

## 3. Ejecución de Pruebas Unitarias

```bash
.venv/bin/pytest tests/ -v
```
