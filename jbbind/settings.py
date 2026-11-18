"""Runtime configuration.

Two layers: immutable process settings from the environment (paths, limits), and a
mutable user-settings document that the Settings page edits and the CLI shares, persisted
as JSON in the cache root so a container restart keeps it.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


#: The hosting server has no GPU, so every prediction runs on the CPU. The model and
#: embedder code still take a ``torch.device``; only the choice was removed.
DEVICE = torch.device("cpu")


@dataclass
class Settings:
    """Process-level settings, from the environment."""

    models_dir: Path = field(
        default_factory=lambda: Path(os.getenv("JBBIND_MODELS", REPO_ROOT / "models")))
    cache_dir: Path = field(
        default_factory=lambda: Path(os.getenv("JBBIND_CACHE", "/data/cache")))
    static_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / "static")

    max_upload_bytes: int = int(os.getenv("JBBIND_MAX_UPLOAD", 30 << 20))
    max_residues: int = int(os.getenv("JBBIND_MAX_RESIDUES", 5000))
    job_workers: int = int(os.getenv("JBBIND_JOB_WORKERS", 2))
    esm_cache_bytes: int = int(os.getenv("JBBIND_ESM_CACHE_BYTES", 20 << 30))
    chain_cache_bytes: int = int(os.getenv("JBBIND_CHAIN_CACHE_BYTES", 10 << 30))

    def __post_init__(self) -> None:
        self.models_dir = Path(self.models_dir)
        self.cache_dir = Path(self.cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Read-only mount (common under Apptainer without a --bind); fall back so the
            # process still starts and /readyz can report the real problem.
            self.cache_dir = Path(os.getenv("TMPDIR", "/tmp")) / "jbbind-cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def device(self) -> torch.device:
        return DEVICE


@dataclass
class UserSettings:
    """User-editable defaults, shared by the web UI and the CLI."""

    # Model
    arch: str = "gnn_mlp"
    setup: str = "protein_nucleic"

    # Inference
    esm_long_seq_mode: str = "tile"       # tile (whole chain) | truncate (training parity)
    rcsb_assembly: int | None = None      # None = asymmetric unit

    # Decision
    threshold: float = 0.5
    per_label_thresholds: dict[str, float] = field(default_factory=dict)

    # Neighbourhood refinement (post-training; see core/refine.py). Off by default:
    # every reported metric describes the raw scores.
    refine_method: str = "none"           # none | bayes | gumbel
    refine_strength: float = 0.5          # beta (bayes) or lambda (gumbel), 0-1
    gumbel_tau: float = 0.5               # softmax temperature
    gumbel_samples: int = 64

    # Display
    color_mode: str = "continuous"        # continuous | threshold
    show_surface: bool = False

    def merged_threshold(self, label: str) -> float:
        return float(self.per_label_thresholds.get(label, self.threshold))


class UserSettingsStore:
    """Thread-safe JSON-backed store for :class:`UserSettings`."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._value = self._read()

    def _read(self) -> UserSettings:
        if not self.path.exists():
            return UserSettings()
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return UserSettings()
        known = {f for f in UserSettings.__dataclass_fields__}
        value = UserSettings(**{k: v for k, v in raw.items() if k in known})
        # A file saved before an architecture was removed (the structure-only
        # "gnn") would otherwise fail every prediction until someone re-saved.
        from .core.nn.registry import ARCHS
        if value.arch not in ARCHS:
            value.arch = UserSettings.arch
        return value

    def get(self) -> UserSettings:
        with self._lock:
            return UserSettings(**asdict(self._value))

    def update(self, patch: dict) -> UserSettings:
        with self._lock:
            current = asdict(self._value)
            known = set(UserSettings.__dataclass_fields__)
            unknown = set(patch) - known
            if unknown:
                raise ValueError(f"unknown settings: {sorted(unknown)}")
            current.update(patch)
            value = UserSettings(**current)
            self._validate(value)
            self._value = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(value), indent=2, default=str))
            os.replace(tmp, self.path)
            return UserSettings(**asdict(value))

    @staticmethod
    def _validate(v: UserSettings) -> None:
        from .core.nn.registry import ARCHS
        from .core.nn.setups import SETUPS
        if v.arch not in ARCHS:
            raise ValueError(f"arch must be one of {sorted(ARCHS)}")
        if v.setup not in SETUPS:
            raise ValueError(f"setup must be one of {sorted(SETUPS)}")
        if v.esm_long_seq_mode not in ("truncate", "tile"):
            raise ValueError("esm_long_seq_mode must be truncate or tile")
        if not 0.0 <= v.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        for label, t in v.per_label_thresholds.items():
            if not 0.0 <= float(t) <= 1.0:
                raise ValueError(f"threshold for {label} must be between 0 and 1")
        if v.color_mode not in ("continuous", "threshold"):
            raise ValueError("color_mode must be continuous or threshold")
        validate_refinement(v)


def validate_refinement(v: UserSettings) -> None:
    """The refine fields alone, shared with the per-job refine endpoint."""
    if v.refine_method not in ("none", "bayes", "gumbel"):
        raise ValueError("refine_method must be none, bayes or gumbel")
    if not 0.0 <= float(v.refine_strength) <= 1.0:
        raise ValueError("refine_strength must be between 0 and 1")
    if not 0.05 < float(v.gumbel_tau) <= 5.0:
        raise ValueError("gumbel_tau must be above 0.05 and at most 5")
    if (isinstance(v.gumbel_samples, bool) or not isinstance(v.gumbel_samples, int)
            or not 8 <= v.gumbel_samples <= 1024):
        raise ValueError("gumbel_samples must be a whole number from 8 to 1024")


settings = Settings()
