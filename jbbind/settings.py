"""Runtime configuration.

Two layers: immutable process settings from the environment (paths, device, limits), and a
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


def _resolve_device(requested: str) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Settings:
    """Process-level settings, from the environment."""

    models_dir: Path = field(
        default_factory=lambda: Path(os.getenv("JBBIND_MODELS", REPO_ROOT / "models")))
    cache_dir: Path = field(
        default_factory=lambda: Path(os.getenv("JBBIND_CACHE", "/data/cache")))
    static_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / "static")

    device_request: str = field(default_factory=lambda: os.getenv("JBBIND_DEVICE", "auto"))

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
        return _resolve_device(self.device_request)


@dataclass
class UserSettings:
    """User-editable defaults, shared by the web UI and the CLI."""

    # Model
    arch: str = "gnn_mlp"
    setup: str = "protein_nucleic"

    # Inference
    device: str = "auto"                  # auto | cpu | cuda
    esm_long_seq_mode: str = "truncate"   # truncate (training parity) | tile
    rcsb_assembly: int | None = None      # None = asymmetric unit

    # Decision
    threshold: float = 0.5
    per_label_thresholds: dict[str, float] = field(default_factory=dict)

    # Display
    color_mode: str = "continuous"        # continuous | threshold
    show_surface: bool = False
    show_sidechains: bool = True

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
        return UserSettings(**{k: v for k, v in raw.items() if k in known})

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
        if v.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if v.esm_long_seq_mode not in ("truncate", "tile"):
            raise ValueError("esm_long_seq_mode must be truncate or tile")
        if not 0.0 <= v.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        for label, t in v.per_label_thresholds.items():
            if not 0.0 <= float(t) <= 1.0:
                raise ValueError(f"threshold for {label} must be between 0 and 1")
        if v.color_mode not in ("continuous", "threshold"):
            raise ValueError("color_mode must be continuous or threshold")


settings = Settings()
