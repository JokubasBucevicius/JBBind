#!/usr/bin/env python3
"""JBBind command line.

    jbbind serve                                   start the web app
    jbbind predict 1ycr --chain A                  one chain to stdout / a CSV
    jbbind batch targets.csv --out results/         many chains, resumable
    jbbind info                                    what is installed and where
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .settings import Settings, UserSettings, UserSettingsStore


def _build(cfg: Settings, overrides: dict | None = None):
    """Wire the pipeline outside the web app."""
    from .core.cache import CacheSet
    from .core.esm.embedder import EsmEmbedder
    from .core.nn.registry import ModelRegistry
    from .core.pipeline import Pipeline

    caches = CacheSet(cfg.cache_dir, esm_max_bytes=cfg.esm_cache_bytes,
                      chain_max_bytes=cfg.chain_cache_bytes)
    store = UserSettingsStore(cfg.cache_dir / "settings.json")
    user = store.get()
    if overrides:
        d = asdict(user)
        d.update({k: v for k, v in overrides.items() if v is not None})
        user = UserSettings(**d)
    registry = ModelRegistry(cfg.models_dir, cfg.device)
    embedder = EsmEmbedder(cfg.device, cache=caches.esm,
                           long_seq_mode=user.esm_long_seq_mode)
    return Pipeline(cfg, registry, embedder, caches), user, registry, caches


def cmd_serve(args) -> int:
    import uvicorn
    uvicorn.run("jbbind.main:app", host=args.host, port=args.port,
                workers=1, log_level=args.log_level)
    return 0


def cmd_info(args) -> int:
    import torch
    cfg = Settings()
    _, user, registry, caches = _build(cfg)
    print(f"device        {cfg.device}  (cuda available: {torch.cuda.is_available()})")
    print(f"models        {cfg.models_dir}")
    print(f"cache         {cfg.cache_dir}")
    try:
        from .core.features.voronota import voronota_version
        print(f"voronota      ok ({voronota_version()})")
    except Exception as exc:
        print(f"voronota      MISSING — {exc}")
    print(f"settings      arch={user.arch} setup={user.setup} "
          f"threshold={user.threshold} esm={user.esm_long_seq_mode}")
    print("\ninstalled checkpoints:")
    for m in registry.available():
        pr = m["metrics"].get("PR AUC (macro)")
        print(f"  {m['setup']:<16} {m['arch']:<8} {'/'.join(m['label_names']):<22} "
              f"PR-AUC {pr if pr is None else round(pr, 3)}")
    print("\ncaches:")
    for c in caches.all_stats():
        print(f"  {c['namespace']:<7} {c['entries']:>7} entries  "
              f"{c['bytes']/1e6:>9.1f} MB")
    return 0


def cmd_predict(args) -> int:
    cfg = Settings()
    pipeline, user, _, _ = _build(cfg, {"arch": args.arch, "setup": args.setup})

    target = args.target
    if Path(target).exists():
        raw, sid, source = pipeline.load_structure(data=Path(target).read_bytes())
        source = f"file {Path(target).name}"
    else:
        raw, sid, source = pipeline.load_structure(pdb_id=target,
                                                   assembly=user.rcsb_assembly)

    chains, _ = pipeline.describe_structure(raw)
    if args.chain:
        wanted = [args.chain]
    elif args.all_chains:
        wanted = [c.chain_id for c in chains]
    else:
        wanted = [chains[0].chain_id]

    from .core.artifacts import predictions_csv

    for chain_id in wanted:
        t0 = time.perf_counter()
        result = pipeline.predict(raw=raw, structure_id=sid, source=source,
                                  chain_id=chain_id, user=user,
                                  setups=[args.setup] if args.setup else None,
                                  progress=(lambda s, m: print(f"  [{s}] {m}",
                                                               file=sys.stderr))
                                  if args.verbose else None)
        elapsed = time.perf_counter() - t0

        if args.json:
            from .main import serialize
            print(json.dumps(serialize(result, user)))
            continue

        csv = predictions_csv(result, args.setup)
        if args.out:
            out = Path(args.out)
            path = out / f"{target}_{chain_id}.csv" if out.is_dir() else out
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(csv)
            print(f"{source} chain {chain_id}: {result.n_predicted} residues "
                  f"in {elapsed:.1f}s -> {path}", file=sys.stderr)
        else:
            setup = args.setup or user.setup
            labels = result.label_names[setup]
            print(f"# {source} chain {chain_id} · {result.arch} · {setup}", file=sys.stderr)
            top = sorted(result.residues,
                         key=lambda r: -max(r.probs[setup]))[:args.top]
            width = max(len(l) for l in labels)
            print(f"{'residue':<12}" + "".join(f"{l:>{width + 3}}" for l in labels))
            for r in top:
                name = f"{r.one_letter}{r.auth_seq_id}{r.auth_icode}"
                print(f"{name:<12}" +
                      "".join(f"{p:>{width + 3}.3f}" for p in r.probs[setup]))
        for w in result.warnings:
            print(f"warning [{w['code']}] {w['detail']}", file=sys.stderr)
    return 0


def cmd_batch(args) -> int:
    from .core.batch import BatchRunner, read_targets

    cfg = Settings()
    pipeline, user, _, _ = _build(cfg, {"arch": args.arch})
    targets = read_targets(Path(args.input))
    setups = [args.setup] if args.setup else None

    runner = BatchRunner(pipeline, cfg, user, Path(args.out),
                         workers=args.workers, setups=setups)
    print(f"{len(targets)} target(s) -> {args.out}  "
          f"(arch={user.arch}, workers={args.workers}, device={cfg.device})")

    last = [0.0]

    def report(p) -> None:
        if time.time() - last[0] < 1.0 and p.done < p.total_targets:
            return
        last[0] = time.time()
        rate = p.done / max(1e-9, p.elapsed)
        eta = (p.total_targets - p.done) / rate if rate else 0
        print(f"\r  {p.done}/{p.total_targets} entries · "
              f"{p.chains_written} chains · {p.failed} failed · "
              f"{rate:.2f}/s · eta {eta/60:.1f}m   ", end="", file=sys.stderr, flush=True)

    progress = runner.run(targets, on_progress=report)
    print(file=sys.stderr)
    print(f"done: {progress.chains_written} chains written, "
          f"{progress.chains_skipped} already complete, {progress.failed} failed, "
          f"{progress.elapsed/60:.1f} min")
    print(f"  {runner.out_dir/'chains'}/            per-chain CSVs")
    print(f"  {runner.out_dir/'predictions.parquet'}  everything, tidy")
    if progress.failed:
        print(f"  {runner.failures_path}          failures")
    return 0 if not progress.failed else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="jbbind", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the web application")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--log-level", default="info")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("predict", help="predict one structure")
    s.add_argument("target", help="4-character PDB ID, or a path to a PDB/mmCIF file")
    s.add_argument("--chain", help="chain to predict (default: the first protein chain)")
    s.add_argument("--all-chains", action="store_true")
    s.add_argument("--setup", help="restrict to one label setup")
    s.add_argument("--arch", help="override the configured architecture")
    s.add_argument("--out", help="write CSV here (a directory, or a file path)")
    s.add_argument("--json", action="store_true", help="emit the full JSON result")
    s.add_argument("--top", type=int, default=20, help="rows to print (default 20)")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("batch", help="predict many structures, resumably")
    s.add_argument("input", help="CSV/TSV/list of `pdb_id[,chain]` or file paths")
    s.add_argument("--out", required=True, help="output directory")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--setup", help="restrict to one label setup")
    s.add_argument("--arch", help="override the configured architecture")
    s.set_defaults(func=cmd_batch)

    sub.add_parser("info", help="show configuration and installed checkpoints") \
       .set_defaults(func=cmd_info)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
