# RL Trainer — contexto para agentes

Aplicación Flask para configurar, visualizar y entrenar agentes de reinforcement learning sobre entornos Gymnasium. El workspace comparte **una sola configuración de experimento** (sesión Flask) entre cuatro pestañas.

Idioma de la UI y del código: **inglés**. El README público está en `README.md`.

## Cómo ejecutar

```bash
# Windows (este repo se trabaja así)
.\.venv\Scripts\activate
pip install -r requirements.txt
python .\app.py
```

Abre `http://127.0.0.1:5000`.

- `app.py` arranca con `debug=True` y **`use_reloader=False`** (evita fugas de semáforos de Pygame/Gymnasium).
- **Cambio en Python no se recarga solo.** Hay que reiniciar el proceso. Plantillas HTML/CSS/JS sí se leen de disco en cada request.
- Si un endpoint nuevo no existe en el proceso viejo, el frontend puede mostrar `Could not inspect the current agent` (fallo de `GET /api/agent/export`).

Pruebas:

```bash
.\.venv\Scripts\python.exe -m unittest tests.test_visualization_no_learning tests.test_zoo_agent
```

## Arquitectura

```
Browser (templates + static/js)
        │  fetch JSON
        ▼
app.py  — rutas HTML + API REST, sesión Flask (`session["config"]`)
        │
        ▼
src/env_runner.py — runtime en memoria por session id (`_RUNTIMES`)
        │           gym.make(..., render_mode="rgb_array")
        ▼
src/agents/TableAgents.py — MC, SARSA, Q_learning (lo que usa la UI hoy)
```

Dos capas de estado:

| Capa | Dónde | Qué guarda |
|------|--------|------------|
| Config | `session["config"]` | environment, agent, hiperparámetros, `allow_learning` |
| Runtime | `_RUNTIMES[sid]` | env Gymnasium, agente tabular, Q-table, timestep, reward |
| Archivo recordado | `session["saved_agent_filename"]` | último JSON/zip cargado o guardado |

Si cambian `environment` o `agent`, `reset_experiment_runtime` tira el runtime y limpia `saved_agent_filename`. Si solo cambian hiperparámetros, se llama `update_runtime_config` y **se conserva la Q-table**.

`src/` se añade a `sys.path` en `app.py`. Imports de agentes: `from agents.TableAgents import ...`.

Rutas de carpetas salen de `config.toml` vía `src/project_paths.py` (`saved_agents`, `static`, `templates`). Artefactos en `saved_agents/` están en `.gitignore` (se conserva `.gitkeep`).

## Pestañas del workspace

| UI | Ruta Flask | Template |
|----|------------|----------|
| Environment | `/load` | `templates/load.html` |
| Visualization | `/visualize-environment` | `templates/visualize_environment.html` |
| Analysis | `/visualize-q-table` | `templates/agent_analysis.html` |
| Training | `/training` | `templates/training.html` |
| Home | `/` | `templates/index.html` |

Layout común: `templates/base.html`. Badge de experimento: `templates/partials/experiment_context.html` + `static/js/experiment-context.js` (`window.ExperimentContext`). El context bar incluye **Save agent** (diálogo compartido en `save_agent_dialog.html` + `static/js/save-agent.js`).

## Archivos clave

| Archivo | Rol |
|---------|-----|
| `app.py` | Servidor, labels de env/agente, `DEFAULT_CONFIG`, todos los endpoints |
| `src/env_runner.py` | Runtime, encode de observaciones, run episode/action, save/load, training |
| `src/project_paths.py` | Resolución segura de paths (`safe_saved_agent_path`) |
| `src/agents/BaseAgent.py` | Clase `Agent` tabular (`Q`, `policy`, `update_policy`) |
| `src/agents/TableAgents.py` | `MC`, `SARSA`, `Q_learning` — **únicos agentes operativos en la UI** |
| `src/agents/deepQ.py`, `linearQ.py`, `linearPolicy.py`, `agentsPG.py`, `agentAC.py`, `networks.py` | Código DRL/aproximación **no cableado** a Flask |
| `templates/load.html` | Formulario Environment + diálogo Load (JS inline). Apply usa `SaveAgent` |
| `templates/visualize_environment.html` | Render, charts Q/policy, run episode/action |
| `templates/agent_analysis.html` | Tabs Q-table / Policy / Value / Metrics |
| `templates/training.html` | Loop de episodios de entrenamiento |
| `templates/partials/save_agent_dialog.html` | Diálogo overwrite vs new file |
| `static/css/design-system.css` | Tokens y controles (botones, inputs) |
| `static/css/style.css` | Layout del dashboard y páginas |
| `static/js/layout.js` | Sidebar móvil |
| `static/js/save-agent.js` | `window.SaveAgent` (preview, diálogo, POST export). Botón **Save agent** en el context bar |
| `config.toml` | Paths relativos a la raíz del repo |
| `tests/test_visualization_no_learning.py` | Visualization no debe aprender si `allow_learning` es false |

## Configuración por defecto

En `app.py` → `DEFAULT_CONFIG`:

- Environment: `FrozenLake-v1`
- Agent: `Q_learning`
- `learning_rate` 0.1, `exploration_probability` 0.1
- `discount_factor` 0.99, `training_episodes` 100, `max_timesteps` 100
- **`allow_learning`: False** — Visualization corre sin actualizar Q a menos que el usuario lo active

Entornos soportados en el selector: Toy Text (`Blackjack-v1`, `Taxi-v4`, `FrozenLake-v1`, `CliffWalking-v1`) y Classic Control (`Acrobot-v1`, `CartPole-v1`, `MountainCar-v0`, `MountainCarContinuous-v0`, `Pendulum-v1`). Los tabulares **exigen action space Discrete**; continuous (MountainCarContinuous, Pendulum) fallan al crear el agente.

El selector de Environment lista agentes tabulares y los de Stable-Baselines3 (`A2C`, `DDPG`, `DQN`, `PPO`, `SAC`, `TD3`). **Apply configuration** instancia el modelo SB3 (`create_sb3_model`) además de guardar la sesión. **Load Zoo agent** descarga el zip pretrained del Hub (`sb3/{algo}-{env}`) y lo pone en `runtime["sb3_model"]`. Visualization corre episodios/acciones con `model.predict`. Las combinaciones incompatibles se deshabilitan: tabulares solo con observación y acción discretas (Blackjack es Tuple, solo tabular); DQN solo acción Discrete; DDPG/SAC/TD3 solo acción Box; PPO y A2C ambas. Analysis **solo soporta** `MC`, `SARSA`, `Q_learning`. Una sesión vieja con `drl-sb3` se migra a `PPO`.

## API (JSON)

Prefijo `/api/*` se sirve con `Cache-Control: no-store`.

| Método | Endpoint | Uso |
|--------|----------|-----|
| GET/POST | `/api/config` | Leer / aplicar config. POST reinicia runtime si cambia env o agente |
| POST | `/api/config/reset` | Defaults + runtime nuevo |
| GET | `/api/environment/initial` | Frame inicial |
| POST | `/api/environment/run-episode` | Episodio hasta `max_timesteps` |
| POST | `/api/environment/run-action` | Un paso (`action`: `"policy"` o índice) |
| GET | `/api/agent/q-table` | Matriz Q (+ `policy`, `value`, `value_definition`) del agente cacheado |
| GET | `/api/agent/analysis` | Igual que q-table: `q_table`, `policy`, `value`, `action_labels`, `n_states`, `n_actions` |
| POST | `/api/agent/evaluate` | 10 episodios greedy (`ε=0`), sin aprender. Restaura epsilon/alpha/Q. `{ rewards, mean, std, n_episodes }` |
| GET | `/api/agent/export` | Preview de guardado (`available`, `current_filename`, `suggested_filename`, `exists`) |
| POST | `/api/agent/export` | Guarda agente. Body: `{ "mode": "overwrite" \| "new", "filename": "..." }` |
| GET | `/api/agent/saved` | Lista archivos en `saved_agents` |
| POST | `/api/agent/import` | Carga `{ "filename" }` (o upload). Alinea config al metadata del archivo |
| POST | `/api/agent/zoo` | Descarga un agente pretrained del RL Baselines3 Zoo (Hub `sb3/{algo}-{env}`) y lo carga en el runtime SB3 |
| POST | `/api/training/start` | Prepara entrenamiento |
| POST | `/api/training/episode` | Un episodio de training |

Nombres de archivo canónicos:

- Tabular: `{env}_{agent}_q_table.json` (metadata `environment` / `agent` dentro del JSON)
- SB3: `{env}_{agent}_model.zip`
- Archivo nuevo: `{env}_{agent}_2_q_table.json`, `_3`, …

## Flujo Environment → Apply configuration

1. Validar learning rate y ε en `[0, 1]`.
2. `GET /api/agent/export` — si `available`, abrir diálogo **Save current agent**.
3. Usuario elige overwrite (archivo actual o default) o new file (nombre editable).
4. `POST /api/agent/export` con esa elección.
5. `POST /api/config` con el formulario.
6. Cancelar el diálogo **no** aplica la config (`Apply cancelled.`).

`Load configuration` lista `saved_agents` y hace `POST /api/agent/import`.

**Load Zoo agent** (solo algoritmos Stable-Baselines3 compatibles con el entorno) abre un diálogo con algorithm, environment, Hub repo (`sb3/{algo}-{env}`) y organization `sb3`, luego `POST /api/agent/zoo`. Eso aplica la config del formulario, descarga el zip del Hub (misma convención que `python -m rl_zoo3.load_from_hub --algo --env -orga sb3`) y sustituye el modelo SB3 en memoria. Si el Hub no tiene ese par, responde 404. Visualization puede “enjoy” el agente con `model.predict`. Apply configuration sigue creando un modelo **sin entrenar**.

El botón **Save agent** del context bar (todas las pestañas del workspace) usa el mismo diálogo/`SaveAgent`. Si el preview dice `available: false`, muestra error. Cancelar no guarda. Apply configuration sigue usando el mismo flujo compartido.

## Convenciones al cambiar UI

- Tokens en `design-system.css`; layout de página en `style.css`.
- Diálogos Environment: clase `.setup-dialog` (también Load configuration).
- Verificar en el navegador el flujo tocado (Apply, Load, Visualization, Training), no solo un screenshot.
- Tras editar `app.py` / `env_runner.py`, **reiniciar Flask**.

## Dependencias

`requirements.txt`: Flask, gymnasium\[classic-control\], numpy, Pillow, stable-baselines3, huggingface_hub.

`src/agents/deepQ.py` importa `torch`; **no está en requirements** porque esa ruta no está conectada a la app.

## Qué no está hecho / limitaciones

- DRL SB3: Apply instancia un modelo sin entrenar; **Load Zoo agent** baja un checkpoint pretrained del Hub. Visualization corre episodios/acciones con `model.predict`. No hay entrenamiento SB3 en la pestaña Training.
- Analysis: Q-table, policy (`agent.policy`), V(s)=max_a Q(s,a), y metrics (10 episodios greedy). DRL sigue sin soporte.
- Agentes en `src/agents/` distintos de TableAgents no se instancian desde Flask.
- Runtime vive en memoria del proceso: reiniciar el servidor pierde la Q-table (salvo que se haya guardado en `saved_agents`).
