#!/usr/bin/env python3
"""JBBind — per-residue binding-site prediction from a structure.

A single-command front end in the shape the published tools use: give it a PDB chain,
get scores, pictures, and an interactive report that opens in your browser.

    python predict_bindingsites.py 1ycr_A
    python predict_bindingsites.py 1ycr --chain A --setup nucleic
    python predict_bindingsites.py 6lu7 --all-chains --setup all
    python predict_bindingsites.py my_model.pdb --chain A
    python predict_bindingsites.py --list targets.txt

Each chain gets its own folder under --out (default ``predictions/``):

    predictions/1ycr_A/
        report_1ycr_A.html                     interactive Mol* report  <- opens
        predictions_1ycr_A.csv                 every residue, every requested label
        annotated_1ycr_A_protein_Protein.pdb   score in the B-factor column
        1ycr_A_protein_Protein.png             the figure
        1ycr_A_protein_Protein.pml             PyMOL session script
        1ycr_A_protein_Protein.cxc             ChimeraX session script
    predictions/_assets/                       Mol*, copied once, shared by every report

This is a thin wrapper. Every number it prints comes from ``jbbind.core.pipeline``, the
same code path the web app and the test suite exercise — the point of the script is the
interface, not a second implementation.

Only the ``gnn_mlp`` architecture is served here. It is the default checkpoint set and the
one the benchmarks were run with; ``--arch`` exists but is deliberately undocumented in
the examples above.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import re
import socketserver
import sys
import time
import webbrowser
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from jbbind.core.artifacts import predictions_csv, predictions_pdb, slug
from jbbind.core.figure import make_figure
from jbbind.core.nn.setups import SETUPS
from jbbind.core.report import write_report
from jbbind.core.viewers import chimerax_script, pymol_script
from jbbind.settings import Settings, UserSettings, UserSettingsStore

# --------------------------------------------------------------------------- targets

def parse_target(spec: str, chain_arg: str | None) -> tuple[str, str | None]:
    """``1ycr_A`` / ``1ycr`` / ``path/to/file.pdb`` -> (target, chain or None).

    An explicit --chain always wins, so ``--list`` files and --chain can be mixed without
    the file silently overriding the flag.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("empty target")
    if Path(spec).exists():
        return spec, chain_arg
    m = re.fullmatch(r"([0-9a-zA-Z]{4})[_:.]([A-Za-z0-9]{1,4})", spec)
    if m:
        return m.group(1).lower(), chain_arg or m.group(2)
    if re.fullmatch(r"[0-9a-zA-Z]{4}", spec):
        return spec.lower(), chain_arg
    raise ValueError(
        f"cannot read {spec!r} as a PDB ID, a <pdb>_<chain> pair, or an existing file")


def read_target_list(path: Path, chain_arg: str | None) -> list[tuple[str, str | None]]:
    out = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "," in line:
            pdb, _, chain = line.partition(",")
            out.append(parse_target(pdb.strip(), chain.strip() or chain_arg))
        else:
            out.append(parse_target(line, chain_arg))
    return out


# --------------------------------------------------------------------------- opening

class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler without the per-request logging."""

    def log_message(self, *args):
        pass


def needs_http() -> bool:
    """Whether a ``file://`` URL would be useless here.

    On a workstation it is fine. On a remote host it is not: ``$BROWSER`` under
    VS Code Remote is a helper that runs ``code --openExternal``, which opens
    the URL on *your laptop*, where ``/home/you/predictions/...`` does not
    exist. Serving over ``http://127.0.0.1`` works instead — VS Code forwards
    the port automatically, and ``ssh -L`` reaches it too.
    """
    if sys.platform in ("darwin", "win32"):
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def open_file(report: Path) -> None:
    if not webbrowser.open(report.resolve().as_uri()):
        print(f"      no browser here — open {report} yourself", file=sys.stderr)


def serve_and_open(root: Path, reports: list[Path], port: int) -> None:
    """Serve ``root`` on localhost, open the report, and block until Ctrl+C.

    Blocking is the point: the page fetches Mol* from ``_assets/`` on load, and
    a reload needs the server still there. A batch run opens the directory
    listing rather than one tab per chain.
    """
    root = root.resolve()
    handler = functools.partial(_QuietHandler, directory=str(root))

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    try:
        httpd = Server(("127.0.0.1", port), handler)
    except OSError as exc:
        print(f"      could not bind 127.0.0.1:{port} ({exc}) — "
              f"open {reports[0]} yourself", file=sys.stderr)
        return

    path = (reports[0].resolve().relative_to(root).as_posix()
            if len(reports) == 1 else "")
    # The bound port, not the requested one: --port 0 means "pick a free one".
    url = f"http://127.0.0.1:{httpd.server_address[1]}/{path}"

    bound = httpd.server_address[1]
    print(f"\n  serving {root}/ at {url}")
    if not webbrowser.open(url):
        print("  (no browser handler here — open that URL yourself; over plain SSH, "
              f"forward it first: ssh -L {bound}:127.0.0.1:{bound} <host>)")
    print("  Ctrl+C to stop the server. The page needs it until it has loaded.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        httpd.server_close()


# --------------------------------------------------------------------------- driver

def build(cfg: Settings, arch: str, threshold: float):
    from jbbind.core.cache import CacheSet
    from jbbind.core.esm.embedder import EsmEmbedder
    from jbbind.core.nn.registry import ModelRegistry
    from jbbind.core.pipeline import Pipeline

    caches = CacheSet(cfg.cache_dir, esm_max_bytes=cfg.esm_cache_bytes,
                      chain_max_bytes=cfg.chain_cache_bytes)
    stored = UserSettingsStore(cfg.cache_dir / "settings.json").get()
    d = asdict(stored)
    d.update(arch=arch, threshold=threshold)
    user = UserSettings(**d)
    registry = ModelRegistry(cfg.models_dir, cfg.device)
    embedder = EsmEmbedder(cfg.device, cache=caches.esm,
                           long_seq_mode=user.esm_long_seq_mode)
    return Pipeline(cfg, registry, embedder, caches), user


def run_chain(pipeline, user, raw, sid, source, chain_id, name, setups, out_root,
              threshold, no_figures, no_report, standalone, verbose) -> dict:
    t0 = time.perf_counter()
    result = pipeline.predict(
        raw=raw, structure_id=sid, source=source, chain_id=chain_id, user=user,
        setups=setups,
        progress=(lambda s, m: print(f"    [{s}] {m}", file=sys.stderr))
        if verbose else None)
    elapsed = time.perf_counter() - t0

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / f"predictions_{name}.csv"]
    written[0].write_text(predictions_csv(result))

    for setup in setups:
        for i, label in enumerate(result.label_names[setup]):
            tag = f"{slug(setup)}_{slug(label)}"
            pdb_path = out_dir / f"annotated_{name}_{tag}.pdb"
            pdb_path.write_text(predictions_pdb(result, setup, i))
            (out_dir / f"{name}_{tag}.pml").write_text(
                pymol_script(name, pdb_path.name, f"{name}_{tag}.pml", label, threshold))
            (out_dir / f"{name}_{tag}.cxc").write_text(
                chimerax_script(name, pdb_path.name, f"{name}_{tag}.cxc", label,
                                threshold))
            written += [pdb_path, out_dir / f"{name}_{tag}.pml",
                        out_dir / f"{name}_{tag}.cxc"]
            if not no_figures:
                png = out_dir / f"{name}_{tag}.png"
                make_figure(result, setup, i, threshold, png)
                written.append(png)

    report = None
    if not no_report:
        report = write_report(result, setups, threshold, name, out_dir, standalone)
        written.append(report)

    print(f"  {name}: {result.n_predicted} residues scored, "
          f"{len(result.unpredicted)} not predicted, {elapsed:.1f}s -> {out_dir}/")
    for setup in setups:
        for i, label in enumerate(result.label_names[setup]):
            vals = [r.probs[setup][i] for r in result.residues]
            n_hit = sum(v >= threshold for v in vals)
            top = max(vals) if vals else float("nan")
            print(f"      {f'{setup}:{label}':<28} {n_hit:>4} at or above "
                  f"{threshold:g}   max {top:.3f}")
    if report is not None:
        print(f"      report                       {report}")
    for w in result.warnings:
        print(f"      warning [{w['code']}] {w['detail']}", file=sys.stderr)
    return {"name": name, "files": written, "result": result, "report": report}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="predict_bindingsites.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("targets", nargs="*",
                   help="PDB ID, <pdb>_<chain>, or a path to a PDB/mmCIF file")
    p.add_argument("--list", dest="list_file",
                   help="file of targets, one `pdb_id[,chain]` per line")
    p.add_argument("--chain", help="chain to predict (default: the first protein chain)")
    p.add_argument("--all-chains", action="store_true",
                   help="predict every protein chain in the entry")
    p.add_argument("--setup", default="protein",
                   help="label setup: " + ", ".join(SETUPS) + ", or 'all' "
                        "(default: protein)")
    p.add_argument("--arch", default="gnn_mlp", help=argparse.SUPPRESS)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="decision threshold for the highlighted sites (default 0.5)")
    p.add_argument("--out", default="predictions", help="output directory")
    p.add_argument("--assembly", type=int, default=None,
                   help="fetch this biological assembly instead of the asymmetric unit")
    p.add_argument("--no-figures", action="store_true", help="skip the PNGs")
    p.add_argument("--no-report", action="store_true",
                   help="skip the interactive HTML report")
    p.add_argument("--standalone", action="store_true",
                   help="inline Mol* in each report (~5 MB) instead of sharing "
                        "one copy under <out>/_assets, so a report can be sent on "
                        "its own")
    p.add_argument("--open", dest="open_report", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="open the report in a browser (default: only when a single "
                        "report was written)")
    p.add_argument("--serve", dest="serve", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="serve the reports over http://127.0.0.1 and block, instead "
                        "of opening a file:// URL (default: on when there is no local "
                        "display, because a remote file:// path means nothing to the "
                        "browser on your machine)")
    p.add_argument("--port", type=int, default=8010,
                   help="port for --serve (default 8010; 0 picks a free one)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.setup == "all":
        setups = list(SETUPS)
    elif args.setup in SETUPS:
        setups = [args.setup]
    else:
        p.error(f"unknown setup {args.setup!r}; choose from {', '.join(SETUPS)}, all")

    targets: list[tuple[str, str | None]] = []
    try:
        for t in args.targets:
            targets.append(parse_target(t, args.chain))
        if args.list_file:
            targets += read_target_list(Path(args.list_file), args.chain)
    except ValueError as exc:
        p.error(str(exc))
    if not targets:
        p.error("give at least one target, or --list a file of them")

    cfg = Settings()
    pipeline, user = build(cfg, args.arch, args.threshold)
    if args.assembly is not None:
        user.rcsb_assembly = args.assembly
    out_root = Path(args.out)

    print(f"JBBind · {user.arch} · setups: {', '.join(setups)} · device {cfg.device}")
    print(f"{len(targets)} target(s) -> {out_root}/")

    failed = 0
    reports: list[Path] = []
    for spec, chain in targets:
        try:
            if Path(spec).exists():
                raw, sid, _ = pipeline.load_structure(data=Path(spec).read_bytes())
                source = f"file {Path(spec).name}"
                stem = Path(spec).stem
            else:
                raw, sid, source = pipeline.load_structure(
                    pdb_id=spec, assembly=user.rcsb_assembly)
                stem = spec.lower()

            chains, _ = pipeline.describe_structure(raw)
            if not chains:
                raise RuntimeError(f"{spec}: no protein chain to predict on")
            if chain:
                wanted = [chain]
            elif args.all_chains:
                wanted = [c.chain_id for c in chains]
            else:
                wanted = [chains[0].chain_id]

            for chain_id in wanted:
                done = run_chain(pipeline, user, raw, sid, source, chain_id,
                                 f"{stem}_{chain_id}", setups, out_root,
                                 args.threshold, args.no_figures, args.no_report,
                                 args.standalone, args.verbose)
                if done["report"] is not None:
                    reports.append(done["report"])
        except Exception as exc:
            failed += 1
            code = getattr(exc, "code", None)
            label = f"[{code}] " if code else ""
            print(f"  {spec}{'_' + chain if chain else ''}: FAILED {label}"
                  f"{getattr(exc, 'message', exc)}", file=sys.stderr)
            hint = HINTS.get(code)
            if hint:
                print(f"      {hint}", file=sys.stderr)
            if code is None and args.verbose:
                raise

    # Opening by default is only kind for a single report; a --list run would
    # otherwise spray a hundred tabs.
    want_open = args.open_report if args.open_report is not None else len(reports) == 1
    if want_open and reports:
        if args.serve if args.serve is not None else needs_http():
            serve_and_open(out_root, reports, args.port)
        else:
            for path in reports:
                open_file(path)

    if failed:
        print(f"\n{failed} target(s) failed", file=sys.stderr)
    return 1 if failed else 0


HINTS = {
    "VoronotaMissing":
        "Put voronota-js on PATH: export PATH=\"$PATH:/path/to/voronota/expansion_js\"",
    "PdbNotFound": "Check the ID at https://www.rcsb.org, or pass a local file.",
    "ChainNotFound":
        "Run the same target with --all-chains to see which chains exist. PDB entries and "
        "derived datasets often disagree about the chain letter.",
    "NoPolymerChains": "This entry has no protein chain long enough to predict on.",
    "SequenceMappingFailed":
        "The observed residues could not be aligned to SEQRES, so the ESM alignment would "
        "be a guess.",
    "NoSurfaceResidues": "No solvent-accessible residue in this chain.",
    "TooManyResidues": "Raise JBBIND_MAX_RESIDUES to run this chain anyway.",
    "RcsbUnavailable": "RCSB could not be reached; pass a local file instead.",
}


if __name__ == "__main__":
    raise SystemExit(main())
