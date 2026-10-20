# Especificación del Sistema Sensorial LIDAR 2D y Vector de Observación

## 1. Geometría del Emisor y Distribución de Rayos

El sensor LIDAR 2D se sitúa en el **centro delantero del chasis** del vehículo:

$$\vec{P}_{\text{origen}} = \vec{P}_{\text{coche}} + \left(\frac{L}{2}\right) \hat{u}_{fwd}$$

donde $L$ es la longitud del chasis y $\hat{u}_{fwd} = (\cos\theta, \sin\theta)$ es el vector unitario de dirección del coche.

Desde $\vec{P}_{\text{origen}}$ se emiten 5 rayos con alcance máximo $R_{\max} = 300.0\text{ px}$ (configurable):

| Identificador | Ángulo Relativo ($\Delta\theta$) | Ángulo Global ($\theta_i$) | Propósito |
| :--- | :---: | :---: | :--- |
| `d_0` | $0^\circ$ | $\theta + 0$ | Detección frontal directa |
| `d_45_left` | $+45^\circ$ | $\theta - \frac{\pi}{4}$ *(en coord. pantalla Y-abajo)* / $+\frac{\pi}{4}$ *(local izq.)* | Anticipación de curva izquierda |
| `d_45_right` | $-45^\circ$ | $\theta + \frac{\pi}{4}$ *(en coord. pantalla Y-abajo)* / $-\frac{\pi}{4}$ *(local der.)* | Anticipación de curva derecha |
| `d_90_left` | $+90^\circ$ | $\theta - \frac{\pi}{2}$ *(local izq.)* | Proximidad lateral izquierda |
| `d_90_right` | $-90^\circ$ | $\theta + \frac{\pi}{2}$ *(local der.)* | Proximidad lateral derecha |

> [!NOTE]
> En coordenadas de pantalla 2D donde el eje $X$ apunta a la derecha y el eje $Y$ apunta hacia abajo, un giro hacia la **izquierda** respecto al vector frontal corresponde a restar el ángulo en radianes ($\theta - \alpha$), mientras que un giro a la **derecha** corresponde a sumar el ángulo ($\theta + \alpha$). La función de transformación local-a-global convierte cualquier ángulo relativo $\Delta\theta \in \{0^\circ, +45^\circ, -45^\circ, +90^\circ, -90^\circ\}$ de forma consistente con el sistema de referencia del vehículo.

---

## 2. Cálculo de Intersección con `segment_query`

Para cada rayo $i \in \{1..5\}$:
1. Se calcula el punto final máximo:
   $$\vec{P}_{\text{fin}, i} = \vec{P}_{\text{origen}} + R_{\max} (\cos\theta_i, \sin\theta_i)$$
2. Se ejecuta `space.segment_query_first(origin, end, radius=1.0, shape_filter=...)` filtrando la propia forma física del vehículo emisor para evitar auto-colisiones, pero detectando barreras estáticas y otros coches.
3. Si existe impacto con factor de interpolación $\alpha \in [0, 1]$ o punto de impacto $\vec{P}_{\text{hit}}$:
   $$d_i = \|\vec{P}_{\text{hit}} - \vec{P}_{\text{origen}}\|_2 \in [0, R_{\max}]$$
   En caso contrario, $d_i = R_{\max}$ y `hit = False`.

---

## 3. Vector de Estado Normalizado ($S$)

El método `get_observation() -> np.ndarray` devuelve un arreglo unidimensional `numpy.ndarray` de tipo `float32` con 7 componentes:

$$S = \left[ \frac{d_0}{R_{\max}}, \frac{d_{45^\circ}}{R_{\max}}, \frac{d_{-45^\circ}}{R_{\max}}, \frac{d_{90^\circ}}{R_{\max}}, \frac{d_{-90^\circ}}{R_{\max}}, \frac{v}{v_{\max}}, \frac{\omega}{\omega_{\max}} \right]$$

### Rangos de Normalización:
- **Distancias LIDAR ($\frac{d_i}{R_{\max}}$):** Acotadas estrictamente en $[0.0, 1.0]$. Un valor de $1.0$ indica vía libre hasta $R_{\max}$; un valor cercano a $0.0$ indica colisión inminente.
- **Velocidad longitudinal normalizada ($\frac{v}{v_{\max}}$):** Acotada en $[-1.0, 1.0]$ mediante `np.clip(v / v_max, -1.0, 1.0)`.
- **Velocidad angular normalizada ($\frac{\omega}{\omega_{\max}}$):** Acotada en $[-1.0, 1.0]$ mediante `np.clip(omega / omega_max, -1.0, 1.0)`.

---

## 4. Modo de Depuración Visual (`F3`)

Al activar el overlay de depuración con `F3`:
- **Rayos sin obstáculo (`hit == False`):** Línea verde (`#00FF66`) desde $\vec{P}_{\text{origen}}$ hasta $\vec{P}_{\text{fin}}$.
- **Rayos con impacto (`hit == True`):** Línea roja (`#FF3333`) desde $\vec{P}_{\text{origen}}$ hasta $\vec{P}_{\text{hit}}$, más un círculo de impacto resaltado en el punto exacto de colisión.
- **HUD de Telemetría:** Panel semitransparente en pantalla que muestra los 7 valores normalizados de $S$, las distancias en píxeles de cada rayo, la velocidad lineal, velocidad angular, superficie actual y FPS.
