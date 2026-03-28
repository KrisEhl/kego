from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class ClusterConfig:
    ray_address: str
    mlflow_uri: str
    repo_path: str = "~/projects/kego"
    uv_project_dir: str = "~/projects/kego/competitions/playground"
    data_path: str = "/home/kristian/projects/kego/data"
    default_resources: dict = field(default_factory=lambda: {"num_gpus": 0.5})
    heavy_resources: dict = field(default_factory=lambda: {"num_gpus": 1})
    all_resources: dict = field(default_factory=dict)


@dataclass
class CompetitionConfig:
    slug: str
    kaggle_user: str
    enable_gpu: bool
    submit_file: str
    pattern: str
    inference_notebook: str
    checkpoint_dir: str
    primary_metric: str
    training_notebook: str | None = None


@dataclass
class KegoConfig:
    cluster: ClusterConfig
    competition: CompetitionConfig | None
    repo_root: Path
    competition_dir: Path | None


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from start until a .git directory is found."""
    if start is None:
        start = Path.cwd()
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    raise FileNotFoundError(f"No .git directory found starting from {start}")


def find_competition_dir(start: Path | None = None) -> Path | None:
    """Walk up from start to find a kego.toml with a [competition] section."""
    if start is None:
        start = Path.cwd()
    for parent in [start, *start.parents]:
        toml_path = parent / "kego.toml"
        if toml_path.exists():
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
            if "competition" in data:
                return parent
    return None


def load_config(
    repo_root: Path | None = None,
    competition_dir: Path | None = None,
) -> KegoConfig:
    """Load root kego.toml, optionally merged with a competition kego.toml."""
    if repo_root is None:
        repo_root = find_repo_root()

    root_toml = repo_root / "kego.toml"
    with open(root_toml, "rb") as f:
        root = tomllib.load(f)

    c = root["cluster"]
    resources = c.get("resources", {})
    cluster = ClusterConfig(
        ray_address=c["ray_address"],
        mlflow_uri=c["mlflow_uri"],
        repo_path=c.get("repo_path", "~/projects/kego"),
        uv_project_dir=c.get(
            "uv_project_dir", "~/projects/kego/competitions/playground"
        ),
        data_path=c.get("data_path", "/home/kristian/projects/kego/data"),
        default_resources=resources.get("default", {"num_gpus": 0.5}),
        heavy_resources=resources.get("heavy", {"num_gpus": 1}),
        all_resources=resources,
    )

    competition = None
    if competition_dir is None:
        competition_dir = find_competition_dir()

    if competition_dir is not None:
        comp_toml = competition_dir / "kego.toml"
        with open(comp_toml, "rb") as f:
            comp_data = tomllib.load(f)
        comp_raw = comp_data["competition"]
        # Allow competition kego.toml to override specific cluster settings.
        if "cluster" in comp_data:
            comp_cluster = comp_data["cluster"]
            if "uv_project_dir" in comp_cluster:
                cluster.uv_project_dir = comp_cluster["uv_project_dir"]
        competition = CompetitionConfig(**comp_raw)

    return KegoConfig(
        cluster=cluster,
        competition=competition,
        repo_root=repo_root,
        competition_dir=competition_dir,
    )
