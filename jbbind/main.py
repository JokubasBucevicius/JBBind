"""JBBind FastAPI application."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse, Response, StreamingResponse)
from fastapi.staticfiles import StaticFiles

from .core.artifacts import (predictions_csv, predictions_pdb, pymol_selection,
                            slug)
from .core.report import report_html
from .core.viewers import chimerax_script, pymol_script
from .core.cache import CacheSet, sha256_bytes
from .core.esm.embedder import EsmEmbedder
from .core.features.voronota import VoronotaError, voronota_version
from .core.jobs import JobQueue
from .core.nn.registry import ARCH_DISPLAY, ARCHS, ModelRegistry
from .core.nn.setups import SETUPS
from .core.pipeline import Pipeline, PredictionResult
from .core.structure.fetch import FetchError, is_pdb_id
from .core.structure.normalize import NormalizationError
from .settings import Settings, UserSettings, UserSettingsStore, settings

STATUS_BY_CODE = {
    "PdbNotFound": 404, "ChainNotFound": 404,
    "UnsupportedFormat": 415,
    "ParseError": 422, "NoPolymerChains": 422, "ChainTooShort": 422,
    "SequenceMappingFailed": 422, "NoSurfaceResidues": 422,
    "PayloadTooLarge": 413, "TooManyResidues": 413,
    "VoronotaFailed": 500, "VoronotaTimeout": 504, "VoronotaMissing": 503,
    "RcsbUnavailable": 502, "Busy": 503,
}


class App:
    """Holds the long-lived objects: registry, embedder, caches, queue."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.caches = CacheSet(cfg.cache_dir, esm_max_bytes=cfg.esm_cache_bytes,
                               chain_max_bytes=cfg.chain_cache_bytes)
        self.registry = ModelRegistry(cfg.models_dir, cfg.device)
        self.embedder = EsmEmbedder(cfg.device, cache=self.caches.esm)
        self.pipeline = Pipeline(cfg, self.registry, self.embedder, self.caches)
        self.jobs = JobQueue(workers=cfg.job_workers)
        self.user_settings = UserSettingsStore(cfg.cache_dir / "settings.json")
        #: uploaded/fetched structures, by content hash
        self.structures: dict[str, tuple[bytes, str]] = {}
        self.results: dict[str, PredictionResult] = {}

    def remember(self, structure_id: str, raw: bytes, source: str) -> None:
        self.structures[structure_id] = (raw, source)
        if len(self.structures) > 64:
            self.structures.pop(next(iter(self.structures)))

    def effective_settings(self, overrides: Optional[dict]) -> UserSettings:
        base = self.user_settings.get()
        if not overrides:
            return base
        d = asdict(base)
        d.update({k: v for k, v in overrides.items() if v is not None})
        return UserSettings(**d)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state: App = app.state.app
    # Load the checkpoints eagerly so /readyz is meaningful and the first request is fast.
    try:
        for setup in SETUPS:
            state.registry.get(setup, state.user_settings.get().arch)
    except Exception as exc:  # surfaced by /readyz
        app.state.model_error = str(exc)
    await asyncio.get_event_loop().run_in_executor(None, state.embedder.load)
    yield
    state.jobs.shutdown()


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or settings
    app = FastAPI(title="JBBind", version="1.0.0", lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.app = App(cfg)
    app.state.model_error = None

    def problem(code: str, message: str, **extra) -> JSONResponse:
        status = STATUS_BY_CODE.get(code, 500)
        body = {"type": f"https://jbbind/errors/{code}", "title": code,
                "status": status, "detail": message, "code": code, **extra}
        return JSONResponse(body, status_code=status,
                            media_type="application/problem+json")

    @app.exception_handler(NormalizationError)
    async def _norm_error(request: Request, exc: NormalizationError):
        return problem(exc.code, exc.message, **exc.extra)

    @app.exception_handler(FetchError)
    async def _fetch_error(request: Request, exc: FetchError):
        return problem(exc.code, exc.message)

    @app.exception_handler(VoronotaError)
    async def _voronota_error(request: Request, exc: VoronotaError):
        return problem(exc.code, exc.message, detail_stderr=exc.stderr or None)

    state: App = app.state.app

    # ---------------------------------------------------------------- health

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        checks = {}
        checks["models"] = {"ok": app.state.model_error is None,
                            "detail": app.state.model_error}
        checks["esm"] = {"ok": state.embedder.loaded}
        try:
            checks["voronota"] = {"ok": True, "version": voronota_version()}
        except Exception as exc:
            checks["voronota"] = {"ok": False, "detail": str(exc)}
        probe = state.cfg.cache_dir / ".write-probe"
        try:
            probe.write_text("ok")
            probe.unlink()
            checks["cache"] = {"ok": True, "path": str(state.cfg.cache_dir)}
        except OSError as exc:
            checks["cache"] = {"ok": False, "detail": str(exc)}
        ok = all(c["ok"] for c in checks.values())
        return JSONResponse({"status": "ready" if ok else "degraded", "checks": checks},
                            status_code=200 if ok else 503)

    @app.get("/api/v1/meta")
    def meta():
        return {
            "version": app.version,
            "device": str(state.cfg.device),
            "cuda_available": torch.cuda.is_available(),
            "setups": {name: {"label_names": s.label_names,
                              "description": s.description, "hint": s.hint}
                       for name, s in SETUPS.items()},
            "archs": list(ARCHS),
            "arch_display": ARCH_DISPLAY,
            "limits": {"max_upload_bytes": state.cfg.max_upload_bytes,
                       "max_residues": state.cfg.max_residues},
        }

    @app.get("/api/v1/models")
    def models():
        return {"models": state.registry.available(),
                "digest": state.registry.manifest_digest()}

    @app.get("/api/v1/metrics")
    def metrics():
        path = state.cfg.models_dir / "METRICS.json"
        if not path.exists():
            raise HTTPException(404, "METRICS.json is missing; run scripts/export_models.py")
        return FileResponse(path, media_type="application/json")

    # -------------------------------------------------------------- settings

    @app.get("/api/v1/settings")
    def get_settings():
        return {"settings": asdict(state.user_settings.get()),
                "caches": state.caches.all_stats()}

    @app.put("/api/v1/settings")
    def put_settings(patch: dict = Body(...)):
        try:
            return {"settings": asdict(state.user_settings.update(patch))}
        except ValueError as exc:
            return problem("InvalidSettings", str(exc))

    @app.post("/api/v1/cache/clear")
    def clear_cache(namespace: str = Query("all")):
        names = ["rcsb", "esm", "chain", "pred"] if namespace == "all" else [namespace]
        removed = {}
        for n in names:
            cache = getattr(state.caches, n, None)
            if cache is None:
                return problem("InvalidSettings", f"unknown cache namespace {n!r}")
            removed[n] = cache.clear()
        return {"removed": removed, "caches": state.caches.all_stats()}

    # ------------------------------------------------------------ structures

    def _structure_info(raw: bytes, structure_id: str, source: str) -> dict:
        chains, warnings = state.pipeline.describe_structure(raw)
        return {
            "structure_id": structure_id,
            "source": source,
            "chains": [asdict(c) for c in chains],
            "warnings": warnings,
        }

    @app.get("/api/v1/structures/by-pdb-id/{pdb_id}")
    def structure_by_pdb_id(pdb_id: str, assembly: Optional[int] = None):
        if not is_pdb_id(pdb_id):
            return problem("UnsupportedFormat", f"{pdb_id!r} is not a 4-character PDB ID")
        raw, sid, source = state.pipeline.load_structure(pdb_id=pdb_id, assembly=assembly)
        state.remember(sid, raw, source)
        return _structure_info(raw, sid, source)

    @app.post("/api/v1/structures")
    async def upload_structure(file: UploadFile = File(...)):
        data = await file.read()
        raw, sid, source = state.pipeline.load_structure(data=data)
        source = f"uploaded {file.filename}"
        state.remember(sid, raw, source)
        return _structure_info(raw, sid, source)

    # --------------------------------------------------------------- predict

    @app.post("/api/v1/predict")
    def predict(body: dict = Body(...)):
        structure_id = body.get("structure_id")
        pdb_id = body.get("pdb_id")
        chain_id = body.get("chain_id")
        overrides = body.get("settings") or {}
        user = state.effective_settings(overrides)

        if structure_id and structure_id in state.structures:
            raw, source = state.structures[structure_id]
        elif pdb_id:
            raw, structure_id, source = state.pipeline.load_structure(
                pdb_id=pdb_id, assembly=user.rcsb_assembly)
            state.remember(structure_id, raw, source)
        else:
            return problem("ParseError",
                           "provide either a known structure_id or a pdb_id")

        if not chain_id:
            chains, _ = state.pipeline.describe_structure(raw)
            chain_id = chains[0].chain_id

        def run(progress, job):
            with state.jobs.heavy:
                result = state.pipeline.predict(
                    raw=raw, structure_id=structure_id, source=source,
                    chain_id=chain_id, user=user, progress=progress)
            state.results[result_key(structure_id, chain_id, user)] = result
            state.results[job.id] = result
            return serialize(result, user)

        job = state.jobs.submit("predict", run)
        return JSONResponse({"job_id": job.id, "status": job.status}, status_code=202)

    def result_key(structure_id: str, chain_id: str, user: UserSettings) -> str:
        return f"{structure_id}:{chain_id}:{user.arch}:{user.esm_long_seq_mode}"

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str):
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id}")
        return job.public()

    @app.get("/api/v1/jobs/{job_id}/events")
    async def job_events(job_id: str):
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id}")

        async def stream():
            seen = 0
            while True:
                while seen < len(job.events):
                    yield f"data: {json.dumps(job.events[seen])}\n\n"
                    seen += 1
                if job.status in ("done", "error"):
                    yield f"data: {json.dumps({'stage': job.status, 'final': True})}\n\n"
                    return
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------- artifacts

    def _result_or_404(job_id: str) -> PredictionResult:
        res = state.results.get(job_id)
        if res is None:
            raise HTTPException(404, f"no result for {job_id}")
        return res

    @app.get("/api/v1/artifacts/{job_id}/receptor.pdb")
    def artifact_receptor(job_id: str):
        return PlainTextResponse(_result_or_404(job_id).receptor_pdb,
                                 media_type="chemical/x-pdb")

    @app.get("/api/v1/artifacts/{job_id}/predictions.csv")
    def artifact_csv(job_id: str, setup: Optional[str] = None):
        res = _result_or_404(job_id)
        name = f"jbbind_{res.chain_id}_{setup or 'all'}.csv"
        return PlainTextResponse(
            predictions_csv(res, setup), media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/api/v1/artifacts/{job_id}/predictions.pdb")
    def artifact_pdb(job_id: str, setup: str = Query(...), label: int = 0):
        res = _result_or_404(job_id)
        name = f"jbbind_{res.chain_id}_{setup}_{label}.pdb"
        return PlainTextResponse(
            predictions_pdb(res, setup, label), media_type="chemical/x-pdb",
            headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/api/v1/artifacts/{job_id}/pymol.txt")
    def artifact_pymol(job_id: str, setup: str = Query(...), label: int = 0,
                       threshold: float = 0.5):
        res = _result_or_404(job_id)
        return PlainTextResponse(pymol_selection(res, setup, label, threshold))

    def _session(job_id: str, setup: str, label: int, threshold: float, ext: str):
        """A PyMOL or ChimeraX session, plus the annotated PDB it expects beside it."""
        res = _result_or_404(job_id)
        tag = f"{slug(setup)}_{slug(res.label_names[setup][label])}"
        stem = f"jbbind_{res.chain_id}_{tag}"
        write = pymol_script if ext == "pml" else chimerax_script
        return PlainTextResponse(
            write(stem, f"annotated_{stem}.pdb", f"{stem}.{ext}",
                  res.label_names[setup][label], threshold),
            headers={"Content-Disposition": f'attachment; filename="{stem}.{ext}"'})

    @app.get("/api/v1/artifacts/{job_id}/session.pml")
    def artifact_pml(job_id: str, setup: str = Query(...), label: int = 0,
                     threshold: float = 0.5):
        return _session(job_id, setup, label, threshold, "pml")

    @app.get("/api/v1/artifacts/{job_id}/session.cxc")
    def artifact_cxc(job_id: str, setup: str = Query(...), label: int = 0,
                     threshold: float = 0.5):
        return _session(job_id, setup, label, threshold, "cxc")

    @app.get("/api/v1/artifacts/{job_id}/figure.png")
    def artifact_figure(job_id: str, setup: str = Query(...), label: int = 0,
                        threshold: float = 0.5):
        from .core.figure import figure_png     # imports matplotlib; endpoint-only

        res = _result_or_404(job_id)
        tag = f"{slug(setup)}_{slug(res.label_names[setup][label])}"
        name = f"jbbind_{res.chain_id}_{tag}.png"
        return Response(
            figure_png(res, setup, label, threshold), media_type="image/png",
            headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/api/v1/artifacts/{job_id}/report.html")
    def artifact_report(job_id: str, threshold: float = 0.5):
        """The standalone interactive report — Mol* inlined, so one file is enough.

        Every setup the job predicted goes in, not just the displayed one: the
        page has its own task selector and a download should not be narrower
        than the page it came from.
        """
        res = _result_or_404(job_id)
        name = f"{res.structure_id}_{res.chain_id}"
        return HTMLResponse(
            report_html(res, list(res.label_names), threshold, name),
            headers={"Content-Disposition":
                     f'attachment; filename="report_{name}.html"'})

    # ------------------------------------------------------------------ SPA

    if cfg.static_dir.exists():
        app.mount("/static", StaticFiles(directory=cfg.static_dir), name="static")

        @app.get("/")
        def index():
            return FileResponse(cfg.static_dir / "index.html")

    return app


def serialize(result: PredictionResult, user: UserSettings) -> dict:
    return {
        "structure_id": result.structure_id,
        "source": result.source,
        "chain_id": result.chain_id,
        "arch": result.arch,
        "sequence": result.sequence,
        "numbering_source": result.numbering_source,
        "label_names": result.label_names,
        "n_predicted": result.n_predicted,
        "residues": [
            {"i": r.seqres_index, "auth": r.auth_seq_id, "icode": r.auth_icode,
             "chain": r.auth_chain, "aa": r.one_letter, "resname": r.resname,
             "sas": r.sas_area, "p": r.probs}
            for r in result.residues
        ],
        "unpredicted": result.unpredicted,
        "warnings": result.warnings,
        "timings_ms": result.timings_ms,
        "settings": asdict(user),
    }


_app: FastAPI | None = None


def __getattr__(name: str):
    """Construct the ASGI app only when something actually asks for it.

    ``uvicorn jbbind.main:app`` still works, but merely *importing* this module no
    longer builds a registry, opens the cache directories and touches the filesystem.
    That matters in a read-only container, where ``import jbbind.main`` would otherwise
    fail trying to create /data/cache before anything had a chance to configure it —
    and it keeps the API contract tests from paying for a full application boot.
    """
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
