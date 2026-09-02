"""Tests for the predict_bindingsites.py front end.

Only the pure parts: target parsing, the colour ramp and the viewer scripts. The
prediction path itself is already covered by test_parity_*.py — the script is a wrapper
over it and must not grow logic of its own that needs separate coverage.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import predict_bindingsites as pb


class TestParseTarget:
    @pytest.mark.parametrize("spec,pdb,chain", [
        ("1ycr_A", "1ycr", "A"),
        ("1YCR_A", "1ycr", "A"),
        ("1ycr:A", "1ycr", "A"),
        ("1ycr.A", "1ycr", "A"),
        ("8cb2_AAA", "8cb2", "AAA"),   # multi-character chain ids exist
        ("1ycr", "1ycr", None),
        ("6LU7", "6lu7", None),
    ])
    def test_forms(self, spec, pdb, chain):
        assert pb.parse_target(spec, None) == (pdb, chain)

    def test_explicit_chain_wins(self):
        """--chain must override the suffix, or a --list run silently ignores the flag."""
        assert pb.parse_target("1ycr_A", "B") == ("1ycr", "B")

    def test_existing_path_is_a_file(self, tmp_path):
        f = tmp_path / "model.pdb"
        f.write_text("ATOM\n")
        assert pb.parse_target(str(f), "A") == (str(f), "A")

    @pytest.mark.parametrize("spec", ["", "  ", "notapdbid", "1yc", "1ycr_"])
    def test_rejects_junk(self, spec):
        with pytest.raises(ValueError):
            pb.parse_target(spec, None)


class TestTargetList:
    def test_forms_and_comments(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("1ycr,B\n3hdd_A\n# comment\n\n6lu7   # trailing\n")
        assert pb.read_target_list(f, None) == [
            ("1ycr", "B"), ("3hdd", "A"), ("6lu7", None)]

    def test_bare_comma_falls_back_to_flag(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("1ycr,\n")
        assert pb.read_target_list(f, "C") == [("1ycr", "C")]


class TestColourRamp:
    def test_endpoints_match_the_stylesheet(self):
        """The CLI figures and the web viewer must agree on what a score looks like."""
        for hexcolor, t in ((pb.SCORE_STOPS[0], 0.0), (pb.SCORE_STOPS[-1], 1.0)):
            want = np.array([int(hexcolor.lstrip("#")[i:i + 2], 16) / 255.0
                             for i in (0, 2, 4)])
            got = np.array(pb.CMAP(t)[:3])
            assert np.allclose(got, want, atol=2 / 255)

    def test_oklab_roundtrip(self):
        for stop in pb.SCORE_STOPS:
            want = np.array([int(stop.lstrip("#")[i:i + 2], 16) / 255.0
                             for i in (0, 2, 4)])
            got = np.array(pb._oklab_to_srgb(pb._srgb_to_oklab(stop)))
            assert np.allclose(got, want, atol=1e-6)

    def test_monotone_lightness(self):
        """A sequential ramp that brightens anywhere would misread as non-monotone score."""
        lum = [0.2126 * r + 0.7152 * g + 0.0722 * b
               for r, g, b, _ in (pb.CMAP(t) for t in np.linspace(0, 1, 64))]
        assert all(a >= b - 1e-9 for a, b in zip(lum, lum[1:]))


class TestSlug:
    @pytest.mark.parametrize("text,want", [
        ("Nucleic acid", "Nucleic_acid"), ("DNA", "DNA"), ("protein_nucleic",
                                                           "protein_nucleic")])
    def test_filename_safe(self, text, want):
        assert pb.slug(text) == want


class TestViewerScripts:
    def test_pymol_excludes_the_sentinel_from_the_ramp(self):
        s = pb.pymol_script("1ycr_A", "annotated.pdb", "1ycr_A.pml", "Protein", 0.5)
        assert "b < -0.5" in s and "color grey70, unpredicted" in s
        # spectrum must be pinned to 0-100, never left to infer from the data range.
        assert "minimum=0, maximum=100" in s
        # palette entries must be defined names, not hex literals
        assert "set_color jb0" in s and "spectrum b, jb0" in s
        assert "#" not in s.split("spectrum b,")[1].split("\n")[0]

    def test_chimerax_uses_the_atom_attribute_selector(self):
        s = pb.chimerax_script("1ycr_A", "annotated.pdb", "1ycr_A.cxc", "Protein", 0.5)
        assert "@@bfactor>=0" in s          # atom attribute, not the residue-level ::
        assert "@@bfactor>=50" in s
        assert "::bfactor" not in s

    def test_threshold_reaches_both_scripts(self):
        assert "b >= 30" in pb.pymol_script("n", "a.pdb", "s.pml", "Protein", 0.3)
        assert "@@bfactor>=30" in pb.chimerax_script("n", "a.pdb", "s.cxc", "Protein", 0.3)


class TestCoordinates:
    def test_reads_ca_only_and_keeps_seqres_numbering(self):
        pdb = (
            "ATOM      1  N   ARG A   5      28.897  13.608  40.938  1.00 93.06           N\n"
            "ATOM      2  CA  ARG A   5      28.659  14.989  40.521  1.00 93.06           C\n"
            "ATOM      3  CA  THR A   6       1.000   2.000   3.000  1.00 90.61           C\n"
            "TER\n")
        coords = pb.ca_coordinates(pdb)
        assert sorted(coords) == [5, 6]
        assert np.allclose(coords[6], [1.0, 2.0, 3.0])


class TestDeexport:
    """The HTML report inlines static/viewer.js as a classic script.

    Browsers refuse ES modules on file:// URLs, so the report cannot import the
    module the web app imports. Stripping `export` is what lets both drive Mol*
    through one file instead of two copies that drift apart.
    """

    def test_strips_every_export_form(self):
        js = ("export const a = 1;\n"
              "export function f() {}\n"
              "export async function g() {}\n"
              "let untouched = 'export inside a string';\n")
        out = pb.deexport(js)
        assert "\nexport " not in "\n" + out
        assert "const a = 1;" in out and "function f() {}" in out
        assert "async function g() {}" in out
        # Only line-initial keywords are keywords; the rest of the file is data.
        assert "'export inside a string'" in out

    def test_rejects_imports(self):
        with pytest.raises(RuntimeError, match="must not import"):
            pb.deexport('import { x } from "./y.js";\nexport const a = 1;\n')

    def test_the_real_viewer_survives_it(self):
        """A guard on viewer.js itself: an import added there breaks every report."""
        out = pb.deexport(pb.VIEWER_JS.read_text(encoding="utf-8"))
        assert "\nexport " not in "\n" + out
        # The names the report's IIFE hands back must all still be defined.
        for name in ("mount", "loadStructure", "paint", "focusResidue",
                     "resetCamera", "resize", "setBackground", "onHover",
                     "onClick", "isReady"):
            assert re.search(rf"(function|const|let) {name}\b", out), name


class TestScoreHex:
    def test_endpoints_are_the_ramp_stops(self):
        assert pb.score_hex(0.0).lower() == pb.SCORE_STOPS[0].lower()
        assert pb.score_hex(1.0).lower() == pb.SCORE_STOPS[-1].lower()

    def test_clamps_out_of_range(self):
        assert pb.score_hex(-3.0) == pb.score_hex(0.0)
        assert pb.score_hex(9.9) == pb.score_hex(1.0)

    def test_always_six_digit_hex(self):
        """viewer.js parses these with parseInt(hex, 16); a short form would silently shift."""
        for t in np.linspace(0, 1, 33):
            assert re.fullmatch(r"#[0-9a-f]{6}", pb.score_hex(t))


@pytest.fixture
def fake_result():
    """A two-residue result, enough to exercise the whole report writer."""
    from jbbind.core.pipeline import PredictionResult, ResiduePrediction
    return PredictionResult(
        structure_id="sid", source="RCSB 3HDD", chain_id="A", arch="gnn_mlp",
        sequence="RT", numbering_source="seqres",
        label_names={"dna_rna": ["DNA", "RNA"]},
        residues=[
            ResiduePrediction(1, "A", 5, "", "ARG", "R", 290.8,
                              {"dna_rna": [0.931, 0.202]}),
            ResiduePrediction(2, "A", 6, "", "THR", "T", 86.2,
                              {"dna_rna": [0.10, 0.01]}),
        ],
        unpredicted=[{"seqres_index": 3, "reason": "buried"}],
        receptor_pdb="ATOM      1  CA  ARG A   1       0.0   0.0   0.0  1.00  0.00\n",
        warnings=[{"code": "no_seqres", "detail": "approximate"}],
        timings_ms={"esm": 12},
    )


class TestReportData:
    def test_shape_and_precomputed_colours(self, fake_result):
        d = pb.report_data(fake_result, ["dna_rna"], 0.5, "3hdd_A")
        assert d["setups"] == ["dna_rna"]
        assert d["labels"]["dna_rna"] == ["DNA", "RNA"]
        assert d["nPredicted"] == 2 and d["nUnpredicted"] == 1
        r = d["residues"][0]
        assert (r["i"], r["auth"], r["aa"], r["sas"]) == (1, 5, "R", 290.8)
        # The page never interpolates: continuous colours arrive resolved, so the
        # ramp keeps one definition shared with the figure.
        assert r["c"]["dna_rna"] == [pb.score_hex(0.931), pb.score_hex(0.202)]
        assert d["rampLo"] == pb.SCORE_STOPS[0] and d["rampHi"] == pb.SCORE_STOPS[-1]
        assert d["unpredicted"] == pb.UNPREDICTED_COLOR

    def test_is_json_serialisable(self, fake_result):
        """numpy floats reach here easily and json.dumps refuses them."""
        json.dumps(pb.report_data(fake_result, ["dna_rna"], 0.5, "3hdd_A"))

    def test_missing_sasa_becomes_null(self, fake_result):
        fake_result.residues[0].sas_area = float("nan")
        d = pb.report_data(fake_result, ["dna_rna"], 0.5, "3hdd_A")
        assert d["residues"][0]["sas"] is None


class TestWriteReport:
    def test_shared_assets(self, fake_result, tmp_path):
        out_dir = tmp_path / "3hdd_A"
        out_dir.mkdir()
        path = pb.write_report(fake_result, ["dna_rna"], 0.5, "3hdd_A", out_dir, False)
        html = path.read_text(encoding="utf-8")

        assert path.name == "report_3hdd_A.html"
        assert not re.search(r"\{\{[A-Z_]+\}\}", html)   # every placeholder substituted
        assert '<script src="../_assets/molstar.js">' in html
        assert (tmp_path / "_assets" / "molstar.js").exists()
        assert (tmp_path / "_assets" / "molstar.css").exists()
        # Mol* is referenced, not inlined: the report stays small.
        assert len(html) < 1_000_000

    def test_standalone_inlines_molstar(self, fake_result, tmp_path):
        out_dir = tmp_path / "3hdd_A"
        out_dir.mkdir()
        path = pb.write_report(fake_result, ["dna_rna"], 0.5, "3hdd_A", out_dir, True)
        html = path.read_text(encoding="utf-8")
        # Neither check can be a bare substring test: the inlined bundle happens
        # to contain both "{{" and "_assets".
        assert not re.search(r"\{\{[A-Z_]+\}\}", html)
        assert "../_assets/molstar.js" not in html
        assert not (tmp_path / "_assets").exists()
        assert len(html) > 4_000_000

    def test_data_cannot_break_out_of_the_script_block(self, fake_result, tmp_path):
        """A "</script>" anywhere in the payload would end the block early."""
        fake_result.warnings = [{"code": "x", "detail": "</script><b>hi</b>"}]
        out_dir = tmp_path / "3hdd_A"
        out_dir.mkdir()
        html = pb.write_report(fake_result, ["dna_rna"], 0.5, "3hdd_A",
                               out_dir, False).read_text(encoding="utf-8")
        payload = html.split('id="jbbind-data"')[1].split("</script>")[0]
        assert "<\\/script>" in payload
        # and it is still valid JSON once the escape is read back
        raw = payload.split(">", 1)[1]
        assert json.loads(raw)["warnings"][0]["detail"] == "</script><b>hi</b>"

    def test_assets_are_refreshed_when_stale(self, fake_result, tmp_path):
        out_dir = tmp_path / "3hdd_A"
        out_dir.mkdir()
        pb.write_report(fake_result, ["dna_rna"], 0.5, "3hdd_A", out_dir, False)
        stale = tmp_path / "_assets" / "molstar.js"
        stale.write_text("stale")
        pb.write_report(fake_result, ["dna_rna"], 0.5, "3hdd_A", out_dir, False)
        assert stale.stat().st_size == pb.MOLSTAR_JS.stat().st_size


class TestNeedsHttp:
    """Whether a file:// URL is worth trying.

    On a remote host it is not: $BROWSER under VS Code Remote runs
    `code --openExternal`, which opens the URL on the user's laptop, where the
    remote path does not exist.
    """

    def test_headless_linux_wants_http(self, monkeypatch):
        monkeypatch.setattr(pb.sys, "platform", "linux")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert pb.needs_http() is True

    @pytest.mark.parametrize("var", ["DISPLAY", "WAYLAND_DISPLAY"])
    def test_a_local_display_is_enough(self, monkeypatch, var):
        monkeypatch.setattr(pb.sys, "platform", "linux")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv(var, ":0")
        assert pb.needs_http() is False

    @pytest.mark.parametrize("platform", ["darwin", "win32"])
    def test_mac_and_windows_have_no_display_var(self, monkeypatch, platform):
        """Neither sets DISPLAY, and both open file:// perfectly well."""
        monkeypatch.setattr(pb.sys, "platform", platform)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert pb.needs_http() is False


class TestServeAndOpen:
    def test_busy_port_falls_back_to_printing_the_path(self, tmp_path, capsys):
        """A second run while the first still serves must not crash the script."""
        import socket
        report = tmp_path / "x" / "report_x.html"
        report.parent.mkdir()
        report.write_text("<html></html>")
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen(1)
            port = busy.getsockname()[1]
            pb.serve_and_open(tmp_path, [report], port)      # must return, not raise
        err = capsys.readouterr().err
        assert "could not bind" in err and str(report) in err

    def test_url_points_at_the_report_and_a_batch_at_the_listing(self, tmp_path,
                                                                 monkeypatch):
        """One report opens itself; many open the directory listing, not N tabs."""
        opened = []
        monkeypatch.setattr(pb.webbrowser, "open", lambda u: opened.append(u) or True)
        # serve_forever would block, so stop the server as soon as it starts.
        monkeypatch.setattr(pb.socketserver.ThreadingTCPServer, "serve_forever",
                            lambda self, *a, **k: None)

        made = []
        for name in ("a", "b"):
            d = tmp_path / name
            d.mkdir()
            f = d / f"report_{name}.html"
            f.write_text("<html></html>")
            made.append(f)

        pb.serve_and_open(tmp_path, made[:1], 0)
        pb.serve_and_open(tmp_path, made, 0)

        assert opened[0].endswith("/a/report_a.html")
        assert re.fullmatch(r"http://127\.0\.0\.1:\d+/", opened[1])
        # Port 0 asks the OS for a free port; the URL must name the one it gave,
        # not the 0 that was requested.
        for url in opened:
            assert not url.startswith("http://127.0.0.1:0/")
