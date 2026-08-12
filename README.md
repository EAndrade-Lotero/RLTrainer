# Telemetría RL

Dashboard en Flask para visualizar y medir el desempeño de un agente de
aprendizaje por refuerzo (Q-learning tabular) entrenando en un entorno
GridWorld.

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate        # en Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

```bash
python3 app.py
```

Abre http://127.0.0.1:5000 en el navegador.

## Estructura

```
app.py                  # servidor Flask + endpoints de la API
rl/environment.py       # entorno GridWorld
rl/agent.py             # agente Q-learning tabular
rl/trainer.py           # hilo de entrenamiento en segundo plano + métricas
templates/index.html    # interfaz con las dos pestañas
static/css/style.css    # estilos
static/js/main.js       # sondeo del estado, render del entorno y gráficas
```

## Pestañas

- **01 · Entorno**: render en vivo de la cuadrícula (agente, meta,
  obstáculos y rastro), lecturas del episodio actual y controles para
  cambiar el tamaño de la cuadrícula y el número de obstáculos.
- **02 · Agente**: métricas de desempeño (episodios, epsilon, recompensa
  media, mejor recompensa, error TD), gráficas de recompensa/pasos/epsilon
  por episodio, y un mapa de calor de los valores Q aprendidos.

## Cómo extender

- **Otro entorno**: crea una clase con `reset()`, `step(action)` y
  `snapshot()` en `rl/environment.py` (o un módulo nuevo) e inyéctala en
  `Trainer._build`.
- **Otro agente**: implementa `act(state)` y `learn(...)` con la misma
  firma que `QLearningAgent` en `rl/agent.py`.
- **Tiempo real más fino**: si necesitas actualizaciones por debajo de los
  200 ms del sondeo actual, se puede sustituir `fetch` por WebSockets
  (Flask-SocketIO) sin cambiar el resto de la arquitectura.
