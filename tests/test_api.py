"""API contract tests, with the pipeline mocked.

Deliberately does not exercise voronota, ESM or the network — those are covered by the
parity tests and the verification scripts. What is checked here is the shape of the HTTP
surface: status codes, error mapping, artifact rendering, and settings validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jbbind.core.artifacts import (UNPREDICTED_B, predictions_csv, predictions_pdb,
                                   pymol_selection)
from jbbind.core.pipeline import PredictionResult, ResiduePrediction
from jbbind.settings import Settings, UserSettings, UserSettingsStore

MODELS = Path(__file__).parents[1] / "models"

RECEPTOR = "\n".join([
    "ATOM      1  N   ALA A   1      11.000  12.000  13.000  1.00  0.00           N",
    "ATOM      2  CA  ALA A   1      12.000  12.000  13.000  1.00  0.00           C",
    "ATOM      3  N   GLY A   2      14.000  12.000  13.000  1.00  0.00           N",
    "ATOM      4  CA  GLY A   2      15.000  12.000  13.000  1.00  0.00           C",
    "ATOM      5  N   SER A   7      17.000  12.000  13.000  1.00  0.00           N",
    "END",
])


def make_result() -> PredictionResult:
    residues = [
        ResiduePrediction(1, "A", 101, "", "ALA", "A", 55.5,
                          {"protein_nucleic": [0.91, 0.02]}),
        ResiduePrediction(2, "A", 102, "", "GLY", "G", 12.0,
                          {"protein_nucleic": [0.30, 0.05]}),
    ]
    return PredictionResult(
        structure_id="deadbeef", source="test", chain_id="A", arch="gnn_mlp",
        sequence="AGXXXXS", numbering_source="seqres",
        label_names={"protein_nucleic": ["Protein", "Nucleic acid"]},
        residues=residues,
        unpredicted=[{"seqres_index": 7, "auth_seq_id": 107, "one_letter": "S",
                      "reason": "buried"}],
        receptor_pdb=RECEPTOR, warnings=[], timings_ms={"model": 5})


# ----------------------------------------------------------------- artifacts

def test_csv_has_one_row_per_residue():
    csv = predictions_csv(make_result())
    lines = csv.strip().splitlines()
    assert lines[0].startswith("seqres_index,auth_chain,auth_seq_id")
    assert "protein_nucleic:Protein" in lines[0]
    assert len(lines) == 3
    assert lines[1].startswith("1,A,101,")


def test_pdb_carries_scores_in_the_b_factor_column():
    pdb = predictions_pdb(make_result(), "protein_nucleic", 0)
    atoms = [l for l in pdb.splitlines() if l.startswith("ATOM")]
    assert len(atoms) == 5
    # residue 1 -> 0.91 * 100
    assert float(atoms[0][60:66]) == pytest.approx(91.0, abs=0.01)
    assert float(atoms[2][60:66]) == pytest.approx(30.0, abs=0.01)
    # residue 7 was not predicted -> the sentinel, never 0.00
    assert float(atoms[4][60:66]) == pytest.approx(UNPREDICTED_B, abs=0.01)
    assert any("REMARK 100" in l and "uncalibrated" in l for l in pdb.splitlines())


def test_pymol_selection_respects_the_threshold():
    sel = pymol_selection(make_result(), "protein_nucleic", 0, threshold=0.5)
    assert "resi 101" in sel and "102" not in sel
    empty = pymol_selection(make_result(), "protein_nucleic", 0, threshold=0.99)
    assert empty.startswith("#")


# ------------------------------------------------------------------ settings

def test_settings_round_trip(tmp_path):
    store = UserSettingsStore(tmp_path / "settings.json")
    assert store.get().arch == "gnn_mlp"
    store.update({"arch": "joint", "threshold": 0.7})
    assert store.get().arch == "joint"
    assert store.get().threshold == 0.7
    # persisted, not just in memory
    assert UserSettingsStore(tmp_path / "settings.json").get().arch == "joint"


@pytest.mark.parametrize("patch", [
    {"arch": "not-an-arch"},
    {"setup": "nonsense"},
    {"threshold": 1.5},
    {"device": "tpu"},
    {"esm_long_seq_mode": "magic"},
    {"color_mode": "rainbow"},
])
def test_invalid_settings_are_rejected(tmp_path, patch):
    store = UserSettingsStore(tmp_path / "settings.json")
    with pytest.raises(ValueError):
        store.update(patch)


def test_unknown_setting_key_is_rejected(tmp_path):
    store = UserSettingsStore(tmp_path / "settings.json")
    with pytest.raises(ValueError):
        store.update({"totally_made_up": 1})


# ----------------------------------------------------------------- endpoints

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import jbbind.main as main

    cfg = Settings()
    cfg.cache_dir = tmp_path_factory.mktemp("cache")
    cfg.models_dir = MODELS
    app = main.create_app(cfg)

    # The TestClient is built WITHOUT entering its context manager, so the lifespan
    # never runs. That is deliberate: the lifespan loads 2.6 GB of ESM-2 weights and
    # eagerly instantiates every checkpoint, which these contract tests do not need
    # and which would make them unrunnable anywhere the weights are absent.
    state = app.state.app
    result = make_result()
    state.results["testjob"] = result
    return TestClient(app, raise_server_exceptions=False), state, result


def test_healthz_is_cheap(client):
    c, _, _ = client
    r = c.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_meta_lists_setups_and_archs(client):
    c, _, _ = client
    r = c.get("/api/v1/meta")
    assert r.status_code == 200
    body = r.json()
    assert set(body["setups"]) == {"protein_nucleic", "homo_hetero", "protein",
                                   "dna_rna", "nucleic"}
    assert body["archs"][0] == "gnn_mlp"


def test_models_endpoint_reports_all_twenty(client):
    c, _, _ = client
    body = c.get("/api/v1/models").json()
    assert len(body["models"]) == 20
    assert all(m["metrics"] for m in body["models"])


def test_metrics_endpoint_serves_the_dashboard_document(client):
    c, _, _ = client
    body = c.get("/api/v1/metrics").json()
    assert "setups" in body and "protein" in body["setups"]
    assert body["threshold"] == 0.5


def test_bad_pdb_id_is_a_problem_json_415(client):
    c, _, _ = client
    r = c.get("/api/v1/structures/by-pdb-id/not-a-pdb-id")
    assert r.status_code == 415          # UnsupportedFormat
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["code"] == "UnsupportedFormat"


def test_predict_without_a_structure_is_rejected(client):
    c, _, _ = client
    r = c.post("/api/v1/predict", json={})
    assert r.status_code == 422
    assert r.json()["code"] == "ParseError"


def test_unknown_job_is_404(client):
    c, _, _ = client
    assert c.get("/api/v1/jobs/nope").status_code == 404
    assert c.get("/api/v1/artifacts/nope/receptor.pdb").status_code == 404


def test_artifacts_render_from_a_stored_result(client):
    c, _, _ = client
    r = c.get("/api/v1/artifacts/testjob/receptor.pdb")
    assert r.status_code == 200 and "ATOM" in r.text

    r = c.get("/api/v1/artifacts/testjob/predictions.csv")
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    assert r.text.splitlines()[0].startswith("seqres_index")

    r = c.get("/api/v1/artifacts/testjob/predictions.pdb?setup=protein_nucleic&label=0")
    assert r.status_code == 200 and "REMARK 100" in r.text

    r = c.get("/api/v1/artifacts/testjob/pymol.txt?setup=protein_nucleic&label=0")
    assert r.status_code == 200 and "select" in r.text


def test_settings_endpoint_validates(client):
    c, _, _ = client
    assert c.get("/api/v1/settings").json()["settings"]["arch"] in ("gnn_mlp", "joint")
    bad = c.put("/api/v1/settings", json={"arch": "nope"})
    assert bad.status_code == 500 or bad.json()["code"] == "InvalidSettings"


def test_cache_clear_rejects_unknown_namespace(client):
    c, _, _ = client
    r = c.post("/api/v1/cache/clear?namespace=bogus")
    assert r.json()["code"] == "InvalidSettings"


# ------------------------------------------------------- generated artifacts
# Everything predict_bindingsites.py writes is also served here, so the browser
# is a complete front end and nothing needs the command line.

def test_report_is_one_self_contained_file(client):
    c, _, _ = client
    r = c.get("/api/v1/artifacts/testjob/report.html?threshold=0.5")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "attachment" in r.headers["content-disposition"]
    html = r.text
    # Mol* inlined, not linked: the download has to work wherever it lands.
    assert "_assets/molstar.js" not in html
    assert len(html) > 4_000_000
    # and the structure travels with it
    assert "ATOM      2  CA  ALA A   1" in html


def test_report_covers_every_setup_not_just_the_displayed_one(client):
    c, _, res = client
    html = c.get("/api/v1/artifacts/testjob/report.html").text
    payload = html.split('id="jbbind-data"')[1].split("</script>")[0].split(">", 1)[1]
    assert json.loads(payload)["setups"] == list(res.label_names)


def test_figure_is_a_png(client):
    c, _, _ = client
    r = c.get("/api/v1/artifacts/testjob/figure.png"
              "?setup=protein_nucleic&label=0&threshold=0.5")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("ext,marker", [("pml", "set_color jb0"),
                                        ("cxc", "@@bfactor>=50")])
def test_viewer_sessions_are_downloadable(client, ext, marker):
    c, _, _ = client
    r = c.get(f"/api/v1/artifacts/testjob/session.{ext}"
              f"?setup=protein_nucleic&label=0&threshold=0.5")
    assert r.status_code == 200
    assert marker in r.text
    assert r.headers["content-disposition"].endswith(f'.{ext}"')


def test_the_job_json_omits_the_receptor_and_the_endpoint_serves_it(client):
    """The receptor is ~40 KB of PDB text and stays out of the job payload.

    The viewer therefore has to fetch it. Reading it off the result instead --
    which never carried it -- is what left the 3D viewer loading an empty model.
    """
    from jbbind.main import serialize

    c, _, res = client
    assert "receptor_pdb" not in serialize(res, UserSettings())
    assert c.get("/api/v1/artifacts/testjob/receptor.pdb").text.startswith("ATOM")


def test_predict_js_fetches_the_receptor_rather_than_reading_the_result():
    """A guard on the front end, since no Python test can catch this one."""
    js = (Path(__file__).parents[1] / "jbbind/static/predict.js").read_text()
    assert "result.receptor_pdb" not in js
    assert "artifacts/${state.jobId}/receptor.pdb" in js
