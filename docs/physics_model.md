# Modelo Físico 2D Arcade: Dinámica Vehicular Simplificada y Asistida

## 1. Filosofía del Modelo Físico (Arcade y Fácil de Controlar)

El motor físico utiliza **Pymunk (Chipmunk 2D)** para resolver colisiones continuas y consultas espaciales de rayos LIDAR (`segment_query`) en tiempo $O(1)$, pero implementa una **dinámica vehicular arcade asistida** diseñada específicamente para que el control humano con teclado sea inmediato, predecible y divertido:

1. **Respuesta inmediata al acelerador y freno (`drive_force = 1350`, `brake_force = 1150`):** El coche acelera con agilidad y frena con contundencia antes de las curvas.
2. **Freno motor automático al soltar el acelerador:** Cuando el jugador suelta `W`/`Flecha Arriba`, el coche desacelera suavemente por sí solo (`effective_drag * 1.75`), facilitando entrar en las curvas sin necesidad de clavar los frenos.
3. **Giro estable sin sobreviraje parásito:**
   - Al soltar las teclas de dirección (`A`/`D` o `Izquierda`/`Derecha`), la velocidad angular se fija instantáneamente en $\omega = 0$. El coche **nunca se queda girando solo** por inercia rotacional.
   - A altas velocidades, la sensibilidad de giro se suaviza ligeramente (`high_speed_stability`) para mantener la trayectoria recta con facilidad.
4. **Agarre alto en conducción normal y Derrape Arcade con `Espacio`:**
   - En conducción normal, el agarre lateral es muy elevado (`lateral_grip = 0.96`), haciendo que el coche vaya exactamente hacia donde apunta el jugador.
   - Al mantener presionada la barra espaciadora (`Espacio`), el agarre lateral pasa a `handbrake_lateral_grip = 0.55` con un ligero incremento de rotación (`1.18x`), permitiendo realizar derrapes largos y controlados sin trompear.
5. **Colisiones permisivas contra muros (`elasticity = 0.15`, `friction = 0.12`):** Si el jugador roza o golpea una barrera, el chasis se desliza suavemente a lo largo del muro en lugar de rebotar bruscamente o quedarse atascado.

---

## 2. Descomposición Cinemática del Chasis

Sea $\theta$ el ángulo de orientación del vehículo en el espacio global 2D:

- **Vector unitario longitudinal (frontal):** $\hat{u}_{fwd} = (\cos\theta, \sin\theta)$
- **Vector unitario lateral (derecha):** $\hat{u}_{lat} = (-\sin\theta, \cos\theta)$

Las proyecciones de la velocidad global $\vec{v} = (v_x, v_y)$ sobre los ejes del chasis son:

$$v_{fwd} = \vec{v} \cdot \hat{u}_{fwd}, \qquad v_{lat} = \vec{v} \cdot \hat{u}_{lat}$$

---

## 3. Superficies del Circuito (`SURFACE_CATALOG`)

Cada circuito cuenta con tres zonas concéntricas evaluadas en tiempo real:

| Superficie | Tracción | Arrastre (`drag_multiplier`) | Agarre Lateral | Velocidad Máxima | Comportamiento para el Jugador |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Asfalto (`asphalt`)** | `1.00` | `1.00` | `1.00` | `100%` ($480\text{ px/s}$) | Máxima velocidad y agarre total |
| **Piano / Borde (`kerb`)** | `0.92` | `1.35` | `0.95` | `92%` | Permite pisar los vértices de las curvas sin perder el control |
| **Césped (`grass`)** | `0.65` | `2.80` | `0.80` | `65%` | Reduce la velocidad al salirse, pero permite reincorporarse fácilmente al asfalto |
