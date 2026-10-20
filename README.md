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

### Selección de Mapas y Opciones CLI
El simulador incluye **4 circuitos gigantes** ($3200 \times 2200\text{ px}$ a $3400 \times 2400\text{ px}$) con cámara de seguimiento y minimapa en vivo:
- `retro_grand_prix` (Tecla `1`): **Retro Grand Prix** (Clásico amplio equilibrado).
- `neon_speedway` (Tecla `2`): **Super Oval & Chicane Speedway** (Alta velocidad).
- `canyon_drift` (Tecla `3`): **Canyon Drift Circuit** (Curvas enlazadas en S para derrapar).
- `starlight_loop` (Tecla `4`): **Starlight Grand Loop** (Trébol panorámico).

```bash
# Iniciar directamente en un mapa y color específicos
.venv/bin/python3 src/main.py --map canyon_drift --color electric_blue

# Iniciar con el piloto autónomo guiado por LIDAR
.venv/bin/python3 src/main.py --map neon_speedway --controller dummy_ai
```

### Controles en Carrera
| Tecla | Acción |
| :--- | :--- |
| `W` / `Flecha Arriba` | Acelerar |
| `S` / `Flecha Abajo` | Frenar / Reversa |
| `A` / `Flecha Izquierda` | Girar a la izquierda |
| `D` / `Flecha Derecha` | Girar a la derecha |
| `Espacio` | **Freno de mano / Modo Derrape** (derrape arcade controlado) |
| `1`, `2`, `3`, `4` o `M` | **Cambiar de circuito en tiempo real** |
| `R` | **Reiniciar posición del coche** en la línea de salida |
| `F3` | Alternar el **Modo de Depuración Visual LIDAR** y **HUD del Vector $S$** |
| `ESC` | Salir del simulador |

---

## 3. Ejecución de Pruebas Unitarias

```bash
.venv/bin/pytest tests/ -v
```
