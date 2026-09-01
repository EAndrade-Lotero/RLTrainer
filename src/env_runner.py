"""Gymnasium + tabular agent episode runner."""

from __future__ import annotations

import base64
import atexit
import io
import json
import secrets
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import gymnasium as gym
import numpy as np
from PIL import Image
from agents.TableAgents import MC, Q_learning, SARSA
from gymnasium.spaces import Box, Discrete, Tuple

DEFAULT_MAX_TIMESTEPS = 100
DEFAULT_GAMMA = 0.99
DEFAULT_TRAINING_EPISODES = 100


def _max_timesteps(runtime: dict[str, Any]) -> int:
    try:
        value = int(
            (runtime.get("config") or {}).get("max_timesteps", DEFAULT_MAX_TIMESTEPS)
        )
    except (TypeError, ValueError):
        return DEFAULT_MAX_TIMESTEPS
    return max(1, value)

TABULAR_AGENTS = {
    "MC": MC,
    "SARSA": SARSA,
    "Q_learning": Q_learning,
}

# In-memory runtime keyed by Flask session id
_RUNTIMES: dict[str, dict[str, Any]] = {}


def _close_env(env) -> None:
    if env is None:
        return
    try:
        env.close()
    except Exception:
        pass


def ensure_session_id(session) -> str:
    if "sid" not in session:
        session["sid"] = secrets.token_hex(16)
    return session["sid"]


def _copy_learned_state(agent) -> dict[str, Any] | None:
    """Snapshot Q, policy, and visit counts so visualization cannot learn."""
    if agent is None:
        return None
    snapshot: dict[str, Any] = {}
    if hasattr(agent, "Q"):
        snapshot["Q"] = np.array(agent.Q, copy=True)
    if hasattr(agent, "policy"):
        snapshot["policy"] = np.array(agent.policy, copy=True)
    if hasattr(agent, "N"):
        snapshot["N"] = np.array(agent.N, copy=True)
    return snapshot


def _restore_learned_state(agent, snapshot: dict[str, Any] | None) -> None:
    if agent is None or snapshot is None:
        return
    if "Q" in snapshot and hasattr(agent, "Q"):
        q_table = agent.Q
        if isinstance(q_table, np.ndarray) and q_table.shape == snapshot["Q"].shape:
            q_table[...] = snapshot["Q"]
        else:
            agent.Q = np.array(snapshot["Q"], copy=True)
    if "policy" in snapshot and hasattr(agent, "policy"):
        policy = agent.policy
        if isinstance(policy, np.ndarray) and policy.shape == snapshot["policy"].shape:
            policy[...] = snapshot["policy"]
        else:
            agent.policy = np.array(snapshot["policy"], copy=True)
    if "N" in snapshot and hasattr(agent, "N"):
        visits = agent.N
        if isinstance(visits, np.ndarray) and visits.shape == snapshot["N"].shape:
            visits[...] = snapshot["N"]
        else:
            agent.N = np.array(snapshot["N"], copy=True)


@contextmanager
def _without_learning(agent) -> Iterator[None]:
    snapshot = _copy_learned_state(agent)
    try:
        yield
    finally:
        _restore_learned_state(agent, snapshot)


def _learning_guard(agent, allow_learning: bool):
    if allow_learning:
        return nullcontext()
    return _without_learning(agent)


def _apply_update(agent, next_state, reward, done, allow_learning: bool) -> None:
    if allow_learning:
        agent.update(next_state, reward, done)


def _reset_agent_knowledge(agent) -> None:
    """Zero learned values (Q-table / policy) if the agent supports it."""
    if agent is None:
        return
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset()
        return
    if hasattr(agent, "Q"):
        agent.Q = np.zeros_like(np.asarray(agent.Q), dtype=float)


def invalidate_runtime(session) -> None:
    sid = session.get("sid")
    if not sid:
        return
    runtime = _RUNTIMES.pop(sid, None)
    if runtime is None:
        return
    _reset_agent_knowledge(runtime.get("agent"))
    _close_env(runtime.get("env"))


def reset_experiment_runtime(session) -> None:
    """Discard the current environment/agent so the next use starts from scratch."""
    session["runtime_generation"] = int(session.get("runtime_generation", 0)) + 1
    session.modified = True
    invalidate_runtime(session)


def runtime_fingerprint(config: dict, session) -> tuple:
    """Identity fingerprint: environment + agent + generation (not hyperparameters)."""
    return (
        config["environment"],
        config["agent"],
        int(session.get("runtime_generation", 0)),
    )


def get_cached_runtime(session) -> dict[str, Any] | None:
    sid = session.get("sid")
    if not sid:
        return None
    return _RUNTIMES.get(sid)


def update_runtime_config(session, config: dict) -> None:
    """Keep the cached agent and refresh hyperparameters / fingerprint."""
    runtime = get_cached_runtime(session)
    if runtime is None:
        return
    runtime["config"] = dict(config)
    runtime["fingerprint"] = runtime_fingerprint(config, session)
    apply_agent_hyperparameters(runtime, config)


def default_agent_filename(config: dict[str, Any] | None) -> str:
    """Canonical on-disk name for an environment/agent pair."""
    config = config or {}
    env_id = str(config.get("environment", "environment")).replace("/", "-")
    agent_id = config.get("agent", "agent")
    if agent_id == "drl-sb3":
        return f"{env_id}_{agent_id}_model.zip"
    return f"{env_id}_{agent_id}_q_table.json"


def normalize_export_filename(filename: str | None, default_name: str) -> str:
    """Keep a basename and add the default extension when the user omitted one."""
    if not filename or not str(filename).strip():
        return default_name
    name = Path(str(filename).strip()).name
    if not name or name in {".", ".."}:
        return default_name
    if not Path(name).suffix:
        name = f"{name}{Path(default_name).suffix}"
    return name


def unique_saved_agent_filename(preferred: str) -> str:
    """Return `preferred` or the next free `<base>_<n><kind><ext>` variant."""
    from project_paths import safe_saved_agent_path

    path = safe_saved_agent_path(preferred)
    if not path.exists():
        return path.name

    stem = path.stem
    suffix = path.suffix
    kind = ""
    base = stem
    for marker in ("_q_table", "_model"):
        if stem.endswith(marker):
            base = stem[: -len(marker)]
            kind = marker
            break

    index = 2
    while True:
        candidate = f"{base}_{index}{kind}{suffix}"
        if not safe_saved_agent_path(candidate).exists():
            return candidate
        index += 1


def export_agent(runtime: dict[str, Any]) -> tuple[bytes, str, str]:
    """
    Serialize the cached agent.

    Tabular agents export Q/policy JSON (same schema as Agent.save).
    Stable-Baselines3 agents use model.save() → .zip (SB3 convention).
    """
    config = runtime.get("config") or {}
    agent_id = config.get("agent", "agent")
    env_id = str(config.get("environment", "environment")).replace("/", "-")
    agent = runtime.get("agent")
    default_name = default_agent_filename(config)

    if agent_id == "drl-sb3":
        model = runtime.get("sb3_model")
        if model is None:
            raise ValueError("No Stable-Baselines3 model is available to save.")
        # SB3 best practice: model.save(path) writes path.zip
        with tempfile.TemporaryDirectory() as tmp:
            save_path = str(Path(tmp) / f"{env_id}_{agent_id}_model")
            model.save(save_path)
            zip_path = Path(f"{save_path}.zip")
            if not zip_path.is_file():
                zip_path = Path(save_path)
            content = zip_path.read_bytes()
        return content, default_name, "application/zip"

    if agent is None or not hasattr(agent, "Q"):
        raise ValueError("No agent in cache to save.")

    payload = {
        "metadata": {
            "environment": config.get("environment", ""),
            "agent": agent_id,
        },
        "Q": np.asarray(agent.Q, dtype=float).tolist(),
    }
    if hasattr(agent, "policy"):
        payload["policy"] = np.asarray(agent.policy, dtype=float).tolist()

    content = json.dumps(payload, indent=4).encode("utf-8")
    return content, default_name, "application/json"


def _identity_from_filename(filename: str) -> dict[str, str]:
    """Recover environment/agent from the `<env>_<agent>[_n]_<kind>` naming scheme."""
    stem = Path(filename).stem
    for suffix in ("_q_table", "_model"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    known_agents = [*TABULAR_AGENTS, "drl-sb3"]
    for agent_id in sorted(known_agents, key=len, reverse=True):
        marker = f"_{agent_id}"
        index = stem.find(marker)
        if index <= 0:
            continue
        remainder = stem[index + len(marker) :]
        if remainder == "" or remainder.startswith("_"):
            return {"environment": stem[:index], "agent": agent_id}
    return {}


def describe_saved_agent(content: bytes, filename: str = "") -> dict[str, str]:
    """
    Best-effort environment/agent identity for a saved artifact.

    Prefers embedded metadata; falls back to the filename for older exports
    and for SB3 zips, which carry no RL Trainer metadata.
    """
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None

    if isinstance(payload, dict):
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            described = {
                key: str(metadata[key])
                for key in ("environment", "agent")
                if metadata.get(key)
            }
            if described:
                return described

    return _identity_from_filename(filename)


def save_agent_to_disk(
    runtime: dict[str, Any],
    filename: str | None = None,
) -> tuple[Path, str]:
    """
    Persist the cached agent under the configured saved_agents directory.

    Returns (absolute_path, kind_label).
    """
    from project_paths import safe_saved_agent_path, saved_agents_dir

    saved_agents_dir(create=True)
    content, default_name, _mime = export_agent(runtime)
    target = safe_saved_agent_path(filename or default_name)

    config = runtime.get("config") or {}
    agent_id = config.get("agent", "agent")
    if agent_id == "drl-sb3":
        model = runtime.get("sb3_model")
        if model is None:
            raise ValueError("No Stable-Baselines3 model is available to save.")
        # Write directly with SB3 so the zip layout matches library conventions.
        save_stem = target.with_suffix("")
        model.save(str(save_stem))
        zip_path = Path(f"{save_stem}.zip")
        if not zip_path.is_file():
            zip_path = save_stem
        return zip_path.resolve(), "Stable-Baselines3 model"

    target.write_bytes(content)
    return target.resolve(), "Q-table"


def list_saved_agents() -> list[dict[str, str]]:
    """List agent artifacts in the configured saved_agents directory."""
    from project_paths import saved_agents_dir

    directory = saved_agents_dir(create=True)
    entries: list[dict[str, str]] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in {".json", ".zip"}:
            continue
        # SB3 zips carry no metadata, so their identity comes from the filename.
        raw = path.read_bytes() if path.suffix.lower() == ".json" else b""
        described = describe_saved_agent(raw, path.name)
        entries.append(
            {
                "filename": path.name,
                "path": str(path.resolve()),
                "kind": "Stable-Baselines3 model" if path.suffix.lower() == ".zip" else "Q-table",
                "environment": described.get("environment", ""),
                "agent": described.get("agent", ""),
            }
        )
    return entries


def import_agent(runtime: dict[str, Any], content: bytes, filename: str = "") -> str:
    """
    Load a previously exported Q-table (JSON) or SB3 model (.zip) into the cache.

    Returns a short description of what was loaded.
    """
    config = runtime.get("config") or {}
    agent_id = config.get("agent", "agent")

    if agent_id == "drl-sb3":
        model = runtime.get("sb3_model")
        if model is None:
            raise ValueError("No Stable-Baselines3 model is available to load into.")
        # SB3 best practice: Algo.load(path) reads path.zip
        with tempfile.TemporaryDirectory() as tmp:
            base_name = Path(filename).stem if filename else "model"
            save_path = Path(tmp) / base_name
            zip_path = save_path.with_suffix(".zip")
            zip_path.write_bytes(content)
            model_cls = type(model)
            if not hasattr(model_cls, "load"):
                raise ValueError("Stable-Baselines3 model does not support load().")
            runtime["sb3_model"] = model_cls.load(str(save_path))
        return "Stable-Baselines3 model"

    agent = runtime.get("agent")
    if agent is None or not hasattr(agent, "Q"):
        raise ValueError("No tabular agent in cache to load into.")

    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Expected a JSON Q-table file.") from exc

    if "Q" not in payload:
        raise ValueError("JSON file must contain a 'Q' field.")

    q_values = np.asarray(payload["Q"], dtype=float)
    expected = np.asarray(agent.Q)
    if q_values.shape != expected.shape:
        described = describe_saved_agent(content, filename)
        source = described.get("environment") or "another environment"
        raise ValueError(
            f"This file was saved for {source} "
            f"(Q-table {tuple(q_values.shape)}), which does not fit "
            f"{config.get('environment', 'the current environment')} "
            f"(Q-table {tuple(expected.shape)})."
        )

    if hasattr(agent, "reset"):
        agent.reset()
    agent.Q = q_values
    if "policy" in payload and hasattr(agent, "policy"):
        policy = np.asarray(payload["policy"], dtype=float)
        if policy.shape != np.asarray(agent.policy).shape:
            raise ValueError(
                f"Policy shape {tuple(policy.shape)} does not match "
                f"current agent shape {tuple(np.asarray(agent.policy).shape)}."
            )
        agent.policy = policy
    elif hasattr(agent, "update_policy"):
        for state in range(int(agent.nS)):
            agent.update_policy(state)

    return "Q-table"


def load_agent_from_disk(runtime: dict[str, Any], filename: str) -> tuple[str, Path]:
    """Load an agent artifact from the configured saved_agents directory."""
    from project_paths import safe_saved_agent_path

    path = safe_saved_agent_path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"Saved agent not found: {path.name}")
    kind = import_agent(runtime, path.read_bytes(), path.name)
    return kind, path


def close_all_runtimes() -> None:
    for sid in list(_RUNTIMES):
        runtime = _RUNTIMES.pop(sid, None)
        if runtime is not None:
            _close_env(runtime.get("env"))


atexit.register(close_all_runtimes)


def _frame_to_data_url(frame) -> str | None:
    if frame is None:
        return None
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _serialize_observation(observation):
    if isinstance(observation, (np.integer, int)):
        return int(observation)
    if isinstance(observation, (np.floating, float)):
        return float(observation)
    if isinstance(observation, tuple):
        return [_serialize_observation(item) for item in observation]
    if isinstance(observation, np.ndarray):
        return observation.tolist()
    return str(observation)


def observation_space_size(space) -> int:
    if isinstance(space, Discrete):
        return int(space.n)
    if isinstance(space, Tuple):
        size = 1
        for subspace in space.spaces:
            if not isinstance(subspace, Discrete):
                raise ValueError("Only Discrete Tuple observation spaces are supported.")
            size *= int(subspace.n)
        return size
    raise ValueError(
        "Tabular agents require a Discrete (or Discrete Tuple) observation space."
    )


def encode_observation(observation, space) -> int:
    if isinstance(space, Discrete):
        return int(observation)
    if isinstance(space, Tuple):
        index = 0
        multiplier = 1
        for value, subspace in zip(reversed(observation), reversed(space.spaces)):
            index += int(value) * multiplier
            multiplier *= int(subspace.n)
        return index
    raise ValueError(
        "Tabular agents require a Discrete (or Discrete Tuple) observation space."
    )


def _state_q_values(runtime: dict[str, Any], observation: Any = None) -> dict | None:
    """Return Q(s, a) for every action using the same table as /api/agent/q-table."""
    env = runtime.get("env")
    try:
        table = get_q_table(runtime)
    except ValueError:
        return None
    if env is None:
        return None

    q_table = np.asarray(table["q_table"], dtype=float)
    if q_table.ndim != 2:
        return None

    if observation is None:
        agent = runtime.get("agent")
        if not getattr(agent, "states", None):
            return None
        state = int(agent.states[-1])
    elif isinstance(observation, (int, np.integer)):
        state = int(observation)
    else:
        try:
            state = encode_observation(observation, env.observation_space)
        except (ValueError, TypeError, AttributeError):
            return None

    if state < 0 or state >= q_table.shape[0]:
        return None

    values = q_table[state]
    max_q = float(np.max(values))
    greedy_actions = [
        int(action) for action, value in enumerate(values) if float(value) == max_q
    ]
    return {
        "state_index": state,
        "values": [float(value) for value in values],
        "greedy_actions": greedy_actions,
    }


def create_agent(agent_name: str, env: gym.Env, config: dict):
    if agent_name not in TABULAR_AGENTS:
        raise ValueError(
            f"Agent '{agent_name}' is not a tabular agent from TableAgents.py."
        )
    if isinstance(env.action_space, Box):
        raise ValueError("Tabular agents require a Discrete action space.")

    parameters = {
        "nS": observation_space_size(env.observation_space),
        "nA": int(env.action_space.n),
        "gamma": float(config.get("discount_factor", DEFAULT_GAMMA)),
        "epsilon": float(config["exploration_probability"]),
        "alpha": float(config["learning_rate"]),
        "first_visit": True,
    }
    return TABULAR_AGENTS[agent_name](parameters)


def get_or_create_runtime(session, config: dict, need_agent: bool = False) -> dict[str, Any]:
    sid = ensure_session_id(session)
    runtime = _RUNTIMES.get(sid)
    fingerprint = runtime_fingerprint(config, session)

    if runtime is not None and runtime.get("fingerprint") == fingerprint:
        runtime["config"] = dict(config)
        if need_agent and runtime.get("agent") is None:
            runtime["agent"] = create_agent(config["agent"], runtime["env"], config)
        apply_agent_hyperparameters(runtime, config)
        return runtime

    if runtime is not None:
        _reset_agent_knowledge(runtime.get("agent"))
        _close_env(runtime.get("env"))

    env = gym.make(config["environment"], render_mode="rgb_array")
    agent = None
    if need_agent or config["agent"] in TABULAR_AGENTS:
        try:
            agent = create_agent(config["agent"], env, config)
        except ValueError:
            if need_agent:
                _close_env(env)
                raise
            agent = None

    runtime = {
        "env": env,
        "agent": agent,
        "fingerprint": fingerprint,
        "config": dict(config),
        "timestep": 0,
        "accumulated_reward": 0.0,
    }
    _RUNTIMES[sid] = runtime
    return runtime


def reset_environment(runtime: dict[str, Any], seed: int | None = 0) -> dict:
    env = runtime["env"]
    agent = runtime.get("agent")
    observation, info = env.reset(seed=seed)
    if agent is not None:
        agent.restart()
        state = encode_observation(observation, env.observation_space)
        agent.states.append(state)
    runtime["timestep"] = 0
    runtime["accumulated_reward"] = 0.0
    frame = env.render()
    return {
        "observation": _serialize_observation(observation),
        "info": {key: _serialize_observation(value) for key, value in info.items()},
        "image": _frame_to_data_url(frame),
        "timestep": 0,
        "max_timesteps": _max_timesteps(runtime),
        "accumulated_reward": 0.0,
        "q_values": _state_q_values(runtime, observation),
    }


def _resolve_action(runtime: dict[str, Any], action_choice: Any):
    env = runtime["env"]
    agent = runtime.get("agent")

    if action_choice is None or action_choice == "policy":
        if agent is None:
            raise ValueError("Agent policy requires a tabular agent.")
        if not agent.states:
            raise ValueError("Environment has not been initialized.")
        return agent.make_decision()

    if isinstance(env.action_space, Discrete):
        action = int(action_choice)
        if action < 0 or action >= env.action_space.n:
            raise ValueError(f"Action must be in 0..{env.action_space.n - 1}.")
        return action

    if isinstance(env.action_space, Box):
        value = float(action_choice)
        low = float(env.action_space.low.flat[0])
        high = float(env.action_space.high.flat[0])
        if value < low or value > high:
            raise ValueError(f"Action must be in [{low}, {high}].")
        return np.array([value], dtype=np.float32)

    raise ValueError("Unsupported action space.")


def run_single_action(
    runtime: dict[str, Any],
    action_choice: Any = "policy",
    allow_learning: bool = False,
) -> dict:
    """Take one environment step. Updates the Q-table only if allow_learning."""
    env = runtime["env"]
    agent = runtime.get("agent")

    if agent is None:
        raise ValueError("Run an action currently requires a tabular agent.")

    with _learning_guard(agent, allow_learning):
        # Start a fresh episode if needed or if the timestep budget was exhausted.
        if not agent.states or runtime.get("timestep", 0) >= _max_timesteps(runtime):
            reset_environment(runtime, seed=None)

        action = _resolve_action(runtime, action_choice)
        agent.actions.append(
            action if isinstance(action, (int, np.integer)) else int(action)
        )

        observation, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        next_state = encode_observation(observation, env.observation_space)

        _apply_update(agent, next_state, reward, done, allow_learning)
        agent.rewards.append(reward)
        agent.dones.append(done)

        runtime["timestep"] = int(runtime.get("timestep", 0)) + 1
        runtime["accumulated_reward"] = float(
            runtime.get("accumulated_reward", 0.0)
        ) + float(reward)

        frame = env.render()
        image = _frame_to_data_url(frame)
        serialized_observation = _serialize_observation(observation)
        serialized_action = (
            int(action)
            if isinstance(action, (int, np.integer))
            else _serialize_observation(action)
        )
        next_image = None
        next_observation = None
        if done:
            observation, info = env.reset()
            agent.restart()
            state = encode_observation(observation, env.observation_space)
            agent.states.append(state)
            runtime["accumulated_reward"] = 0.0
            next_image = _frame_to_data_url(env.render())
            next_observation = _serialize_observation(observation)
        else:
            agent.states.append(next_state)

    payload = {
        "timestep": runtime["timestep"],
        "max_timesteps": _max_timesteps(runtime),
        "accumulated_reward": runtime["accumulated_reward"],
        "reward": float(reward),
        "done": done,
        "action": serialized_action,
        "image": image,
        "observation": serialized_observation,
        "q_values": _state_q_values(runtime, serialized_observation),
        "reset_after_done": done,
    }
    if done:
        payload["next_image"] = next_image
        payload["next_observation"] = next_observation
        payload["next_q_values"] = _state_q_values(runtime, next_observation)
    return payload


def run_until_max_timesteps(
    runtime: dict[str, Any],
    max_timesteps: int = DEFAULT_MAX_TIMESTEPS,
    action_choice: Any = "policy",
    allow_learning: bool = False,
) -> dict:
    """
    Run up to max_timesteps environment steps.

    Uses the selected action (or the agent policy) at every step.
    When an episode ends (terminated or truncated), restart the env and agent
    episode buffers. The Q-table is updated only if allow_learning is True.
    """
    env = runtime["env"]
    agent = runtime.get("agent")
    if agent is None:
        raise ValueError("No tabular agent is available for this configuration.")

    with _learning_guard(agent, allow_learning):
        initial = reset_environment(runtime, seed=None)
        frames = [
            {
                "timestep": 0,
                "accumulated_reward": 0.0,
                "reward": 0.0,
                "done": False,
                "image": initial["image"],
                "observation": initial["observation"],
                "q_values": None,
            }
        ]

        accumulated_reward = 0.0

        for timestep in range(1, max_timesteps + 1):
            action = _resolve_action(runtime, action_choice)
            agent.actions.append(
                action if isinstance(action, (int, np.integer)) else int(action)
            )

            observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            next_state = encode_observation(observation, env.observation_space)

            _apply_update(agent, next_state, reward, done, allow_learning)
            agent.rewards.append(reward)
            agent.dones.append(done)

            accumulated_reward += float(reward)
            serialized_observation = _serialize_observation(observation)
            frame = env.render()
            frames.append(
                {
                    "timestep": timestep,
                    "accumulated_reward": accumulated_reward,
                    "reward": float(reward),
                    "done": done,
                    "action": int(action)
                    if isinstance(action, (int, np.integer))
                    else serialized_observation,
                    "image": _frame_to_data_url(frame),
                    "observation": serialized_observation,
                    "q_values": None,
                }
            )

            if done:
                observation, info = env.reset()
                agent.restart()
                state = encode_observation(observation, env.observation_space)
                agent.states.append(state)
                accumulated_reward = 0.0
            else:
                agent.states.append(next_state)

    for frame_payload in frames:
        frame_payload["q_values"] = _state_q_values(
            runtime, frame_payload["observation"]
        )

    runtime["timestep"] = max_timesteps
    runtime["accumulated_reward"] = frames[-1]["accumulated_reward"]

    return {
        "frames": frames,
        "timestep": max_timesteps,
        "max_timesteps": max_timesteps,
        "accumulated_reward": frames[-1]["accumulated_reward"],
    }


EVALUATION_EPISODES = 10


def get_q_table(runtime: dict[str, Any]) -> dict:
    """Serialize the tabular agent's Q-table for analysis views (read-only)."""
    return get_analysis(runtime)


def get_analysis(runtime: dict[str, Any]) -> dict:
    """Serialize Q, policy, and V(s) = max_a Q(s, a) for analysis views."""
    agent = runtime.get("agent")
    if agent is None:
        raise ValueError(
            "Q-table analysis requires a tabular agent (MC, SARSA, or Q-learning)."
        )
    if not hasattr(agent, "Q"):
        raise ValueError("The current agent does not expose a Q-table.")

    q_values = np.asarray(agent.Q, dtype=float)
    if q_values.ndim != 2:
        raise ValueError("Unexpected Q-table shape.")

    payload = {
        "n_states": int(q_values.shape[0]),
        "n_actions": int(q_values.shape[1]),
        "q_table": q_values.tolist(),
        "value": np.max(q_values, axis=1).tolist(),
        "value_definition": "V(s) = max_a Q(s, a)",
    }
    if hasattr(agent, "policy"):
        policy = np.asarray(agent.policy, dtype=float)
        if policy.shape == q_values.shape:
            payload["policy"] = policy.tolist()
    return payload


def evaluate_greedy_episodes(
    runtime: dict[str, Any],
    n_episodes: int = EVALUATION_EPISODES,
    max_timesteps: int | None = None,
) -> dict:
    """
    Run greedy evaluation episodes without learning or changing session config.

    Temporarily sets epsilon to 0 and rebuilds the policy, then restores
    Q, policy, epsilon, and alpha.
    """
    env = runtime.get("env")
    agent = runtime.get("agent")
    if env is None or agent is None:
        raise ValueError("Evaluation currently requires a tabular agent.")
    if n_episodes < 1:
        raise ValueError("Evaluation requires at least one episode.")

    horizon = _max_timesteps(runtime) if max_timesteps is None else max(1, int(max_timesteps))
    snapshot = _copy_learned_state(agent)
    previous_epsilon = float(getattr(agent, "epsilon", 0.0))
    previous_alpha = float(agent.alpha) if hasattr(agent, "alpha") else None
    previous_param_epsilon = (agent.parameters or {}).get("epsilon")
    previous_param_alpha = (agent.parameters or {}).get("alpha") if hasattr(agent, "parameters") else None

    try:
        agent.epsilon = 0.0
        if hasattr(agent, "parameters"):
            agent.parameters["epsilon"] = 0.0
        if hasattr(agent, "update_policy"):
            n_states = int(getattr(agent, "nS", 0))
            for state in range(n_states):
                agent.update_policy(state)

        rewards: list[float] = []
        for _ in range(n_episodes):
            observation, _info = env.reset()
            agent.restart()
            state = encode_observation(observation, env.observation_space)
            agent.states.append(state)

            episode_reward = 0.0
            for _step in range(horizon):
                action = agent.make_decision()
                observation, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
                next_state = encode_observation(observation, env.observation_space)
                episode_reward += float(reward)
                if done:
                    break
                agent.states.append(next_state)
            rewards.append(episode_reward)
    finally:
        _restore_learned_state(agent, snapshot)
        agent.epsilon = previous_epsilon
        if hasattr(agent, "parameters"):
            if previous_param_epsilon is not None:
                agent.parameters["epsilon"] = previous_param_epsilon
            else:
                agent.parameters["epsilon"] = previous_epsilon
            if previous_param_alpha is not None:
                agent.parameters["alpha"] = previous_param_alpha
        if previous_alpha is not None:
            agent.alpha = previous_alpha

    rewards_array = np.asarray(rewards, dtype=float)
    return {
        "rewards": [float(value) for value in rewards_array],
        "mean": float(np.mean(rewards_array)) if rewards_array.size else 0.0,
        "std": float(np.std(rewards_array)) if rewards_array.size else 0.0,
        "n_episodes": int(len(rewards)),
        "exploration": 0.0,
        "max_timesteps": horizon,
    }


def apply_agent_hyperparameters(runtime: dict[str, Any], config: dict) -> None:
    """Update in-memory tabular agent hyperparameters from the active config."""
    agent = runtime.get("agent")
    if agent is None:
        return
    agent.epsilon = float(config["exploration_probability"])
    agent.gamma = float(config.get("discount_factor", DEFAULT_GAMMA))
    if hasattr(agent, "alpha"):
        agent.alpha = float(config["learning_rate"])
    agent.parameters["epsilon"] = agent.epsilon
    agent.parameters["gamma"] = agent.gamma
    if "alpha" in agent.parameters:
        agent.parameters["alpha"] = float(config["learning_rate"])


def run_training_episode(
    runtime: dict[str, Any],
    max_timesteps: int = DEFAULT_MAX_TIMESTEPS,
) -> dict:
    """
    Run one training episode with the tabular agent.

    Uses the agent's policy, steps the environment, and updates Q-values.
    """
    env = runtime["env"]
    agent = runtime.get("agent")
    if agent is None:
        raise ValueError("Training currently requires a tabular agent.")

    observation, info = env.reset()
    agent.restart()
    state = encode_observation(observation, env.observation_space)
    agent.states.append(state)

    episode_reward = 0.0
    steps = 0
    done = False

    for _ in range(max_timesteps):
        action = agent.make_decision()
        agent.actions.append(
            action if isinstance(action, (int, np.integer)) else int(action)
        )

        observation, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        next_state = encode_observation(observation, env.observation_space)

        agent.update(next_state, reward, done)
        agent.rewards.append(reward)
        agent.dones.append(done)

        episode_reward += float(reward)
        steps += 1

        if done:
            break
        agent.states.append(next_state)

    runtime["timestep"] = steps
    runtime["accumulated_reward"] = episode_reward

    return {
        "episode_reward": episode_reward,
        "steps": steps,
        "done": done,
        "max_timesteps": max_timesteps,
    }
