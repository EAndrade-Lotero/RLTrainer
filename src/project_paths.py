"""Central path resolution from config.toml (repository-root relative)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore


# Repository root is the parent of src/
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = REPO_ROOT / "config.toml"


@lru_cache(maxsize=1)
def load_paths_config() -> dict[str, str]:
    """Load the [paths] table from config.toml."""
    if not CONFIG_FILE.is_file():
        raise FileNotFoundError(f"Missing path config: {CONFIG_FILE}")
    with CONFIG_FILE.open("rb") as handle:
        data = tomllib.load(handle)
    paths = data.get("paths")
    if not isinstance(paths, dict) or not paths:
        raise ValueError(f"config.toml must define a non-empty [paths] table: {CONFIG_FILE}")
    return {str(key): str(value) for key, value in paths.items()}


def resolve_path(key: str) -> Path:
    """
    Resolve a configured path key to an absolute Path.

    Relative entries are joined to the repository root; absolute entries are kept.
    """
    paths = load_paths_config()
    if key not in paths:
        known = ", ".join(sorted(paths))
        raise KeyError(f"Unknown path key {key!r}. Known keys: {known}")
    raw = Path(paths[key]).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (REPO_ROOT / raw).resolve()


def saved_agents_dir(*, create: bool = True) -> Path:
    """Return the configured saved-agents directory, creating it when requested."""
    path = resolve_path("saved_agents")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def safe_saved_agent_path(filename: str) -> Path:
    """
    Resolve a basename under saved_agents, rejecting path traversal.
    """
    name = Path(filename).name
    if not name or name in {".", ".."}:
        raise ValueError("Invalid saved agent filename.")
    directory = saved_agents_dir(create=True)
    path = (directory / name).resolve()
    if path.parent != directory.resolve():
        raise ValueError("Saved agent path must stay inside the saved_agents directory.")
    return path
