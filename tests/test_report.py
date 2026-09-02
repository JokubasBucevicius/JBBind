"""The standalone interactive HTML report."""

import json
import re

import pytest

from jbbind.core import report
from jbbind.core.colour import SCORE_STOPS, UNPREDICTED_COLOR, score_hex


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
        out = report.deexport(js)
        assert "\nexport " not in "\n" + out
        assert "const a = 1;" in out and "function f() {}" in out
        assert "async function g() {}" in out
        # Only line-initial keywords are keywords; the rest of the file is data.
        assert "'export inside a string'" in out

    def test_rejects_imports(self):
        with pytest.raises(RuntimeError, match="must not import"):
            report.deexport('import { x } from "./y.js";\nexport const a = 1;\n')

    def test_the_real_viewer_survives_it(self):
        """A guard on viewer.js itself: an import added there breaks every report."""
        out = report.deexport(report.VIEWER_JS.read_text(encoding="utf-8"))
        assert "\nexport " not in "\n" + out
        # The names the report's IIFE hands back must all still be defined.
        for name in ("mount", "loadStructure", "paint", "focusResidue",
                     "resetCamera", "resize", "setBackground", "onHover",
                     "onClick", "isReady"):
            assert re.search(rf"(function|const|let) {name}\b", out), name


class TestReportData:
    def test_shape_and_precomputed_colours(self, fake_result):
        d = report.report_data(fake_result, ["dna_rna"], 0.5, "3hdd_A")
        assert d["setups"] == ["dna_rna"]
        assert d["labels"]["dna_rna"] == ["DNA", "RNA"]
        assert d["nPredicted"] == 2 and d["nUnpredicted"] == 1
        r = d["residues"][0]
        assert (r["i"], r["auth"], r["aa"], r["sas"]) == (1, 5, "R", 290.8)
        # The page never interpolates: continuous colours arrive resolved, so the
        # ramp keeps one definition shared with the figure.
        assert r["c"]["dna_rna"] == [score_hex(0.931), score_hex(0.202)]
        assert d["rampLo"] == SCORE_STOPS[0] and d["rampHi"] == SCORE_STOPS[-1]
        assert d["unpredicted"] == UNPREDICTED_COLOR

    def test_is_json_serialisable(self, fake_result):
        """numpy floats reach here easily and json.dumps refuses them."""
        json.dumps(report.report_data(fake_result, ["dna_rna"], 0.5, "3hdd_A"))

    def test_missing_sasa_becomes_null(self, fake_result):
        fake_result.residues[0].sas_area = float("nan")
        d = report.report_data(fake_result, ["dna_rna"], 0.5, "3hdd_A")
        assert d["residues"][0]["sas"] is None


class TestWriteReport:
    def test_shared_assets(self, fake_result, tmp_path):
        out_dir = tmp_path / "3hdd_A"
        out_dir.mkdir()
        path = report.write_report(fake_result, ["dna_rna"], 0.5, "3hdd_A", out_dir, False)
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
        path = report.write_report(fake_result, ["dna_rna"], 0.5, "3hdd_A", out_dir, True)
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
        html = report.write_report(fake_result, ["dna_rna"], 0.5, "3hdd_A",
                               out_dir, False).read_text(encoding="utf-8")
        payload = html.split('id="jbbind-data"')[1].split("</script>")[0]
        assert "<\\/script>" in payload
        # and it is still valid JSON once the escape is read back
        raw = payload.split(">", 1)[1]
        assert json.loads(raw)["warnings"][0]["detail"] == "</script><b>hi</b>"

    def test_assets_are_refreshed_when_stale(self, fake_result, tmp_path):
        out_dir = tmp_path / "3hdd_A"
        out_dir.mkdir()
        report.write_report(fake_result, ["dna_rna"], 0.5, "3hdd_A", out_dir, False)
        stale = tmp_path / "_assets" / "molstar.js"
        stale.write_text("stale")
        report.write_report(fake_result, ["dna_rna"], 0.5, "3hdd_A", out_dir, False)
        assert stale.stat().st_size == report.MOLSTAR_JS.stat().st_size
