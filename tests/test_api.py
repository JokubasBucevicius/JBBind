"""API contract tests, with the pipeline mocked.

Deliberately does not exercise voronota, ESM or the network — those are covered by the
parity tests. What is checked here is the shape of the HTTP surface: status codes, error
mapping, artifact rendering, and settings validation.
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


def test_a_saved_gnn_arch_falls_back_to_the_default(tmp_path):
    """settings.json written before the GNN was removed must not break predictions."""
    (tmp_path / "settings.json").write_text(
        json.dumps({"arch": "gnn", "show_sidechains": True, "threshold": 0.6}))
    s = UserSettingsStore(tmp_path / "settings.json").get()
    assert s.arch == "gnn_mlp" and s.threshold == 0.6


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
    assert body["archs"] == ["gnn_mlp", "joint", "mlp"]
    assert body["setups"]["protein_nucleic"]["label_kinds"] == ["protein", "nucleic"]
    assert body["setups"]["dna_rna"]["label_kinds"] == ["nucleic", "nucleic"]


def test_models_endpoint_reports_all_fifteen(client):
    c, _, _ = client
    body = c.get("/api/v1/models").json()
    assert len(body["models"]) == 15
    assert not any(m["arch"] == "gnn" for m in body["models"])
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


# ---------------------------------------------------------- every label shown
# The score columns are built from the selected setup's label_names, so a task with
# two labels shows two of them. A single hardcoded `score` header would silently
# collapse that back to one and no Python test would notice.

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("page", ["jbbind/static/index.html", "jbbind/report_template.html"])
def test_residue_table_has_no_hardcoded_score_column(page):
    html = (ROOT / page).read_text()
    assert 'data-sort="p"' not in html
    assert '<th data-sort="sas">SASA</th>' in html


@pytest.mark.parametrize("script", ["jbbind/static/predict.js", "jbbind/report_template.html"])
def test_score_columns_are_generated_per_label(script):
    src = (ROOT / script).read_text()
    assert "renderScoreColumns" in src
    assert 'data-label' in src


def test_esm_defaults_to_tiling_so_long_chains_are_scored_end_to_end():
    from jbbind.settings import Settings, UserSettings
    assert UserSettings().esm_long_seq_mode == "tile"
    assert Settings().max_residues >= 5000


# ------------------------------------------------------------- opening cost
# Mol* is 4.8 MB. Reached over an SSH tunnel, anything that puts it in the path
# of first paint leaves the tab on "loading" until the whole bundle arrives.

def test_static_assets_are_compressed(client):
    c, _, _ = client
    r = c.get("/static/vendor/molstar.js", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers.get("content-encoding") == "gzip"


def test_api_responses_are_not_compressed(client):
    """gzip is mounted on /static alone.

    Applying it app-wide would also wrap /api/v1/jobs/{id}/events, where
    buffering stalls the SSE stream the Predict page waits on.
    """
    c, _, _ = client
    r = c.get("/api/v1/meta", headers={"Accept-Encoding": "gzip"})
    assert "content-encoding" not in r.headers


def test_index_does_not_block_first_paint_on_molstar():
    html = (Path(__file__).parents[1] / "jbbind/static/index.html").read_text()
    assert "vendor/molstar.js" not in html          # viewer.js fetches it on first use
    js = (Path(__file__).parents[1] / "jbbind/static/viewer.js").read_text()
    assert "/static/vendor/molstar.js" in js


# ------------------------------------------------------------------- device

def test_server_runs_on_the_cpu_only(client, monkeypatch):
    """The hosting server has no GPU; a visible one must not change that."""
    c, _, _ = client
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert Settings().device.type == "cpu"
    meta = c.get("/api/v1/meta").json()
    assert meta["device"] == "cpu" and "cuda_available" not in meta


def test_device_is_no_longer_a_setting(tmp_path):
    assert "device" not in UserSettings.__dataclass_fields__
    with pytest.raises(ValueError):
        UserSettingsStore(tmp_path / "settings.json").update({"device": "cpu"})


def test_settings_file_from_before_the_removal_still_loads(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"device": "cuda", "threshold": 0.7}))
    assert UserSettingsStore(path).get().threshold == 0.7


def test_old_page_sending_a_device_does_not_break_predict(client):
    """A tab still running the old JS sends settings.device; it is ignored."""
    c, app_state, _ = client
    assert app_state.effective_settings({"device": "cuda", "threshold": 0.3}).threshold == 0.3


def test_predict_page_offers_no_device_choice():
    src = (ROOT / "jbbind/static/predict.js").read_text()
    assert "settings: refineSettings()," in src
    html = (ROOT / "jbbind/static/index.html").read_text()
    assert 'id="device"' not in html and "GPU" not in html
    assert "cuda" not in (ROOT / "jbbind/static/settings.js").read_text()


# ----------------------------------------------------------------- display

@pytest.mark.parametrize("page", ["jbbind/static/index.html", "jbbind/static/predict.js",
                                  "jbbind/static/settings.js", "jbbind/static/viewer.js",
                                  "jbbind/report_template.html"])
def test_side_chain_sticks_are_gone(page):
    src = (ROOT / page).read_text().lower()
    for token in ("show-sidechains", "show_sidechains", "show-sticks", "showsticks",
                  "ball-and-stick"):
        assert token not in src, token


def test_hidden_really_hides():
    """.status sets display:flex, which beats the browser's [hidden] rule; without this
    override the status bar and its spinner never go away after a prediction."""
    css = (ROOT / "jbbind/static/style.css").read_text()
    assert "[hidden] { display: none !important; }" in css


def test_theme_picker_offers_light_dark_and_system():
    html = (ROOT / "jbbind/static/index.html").read_text()
    for value in ("system", "light", "dark"):
        assert f'<option value="{value}">' in html
    # Applied before the stylesheet, so a dark choice does not flash light first.
    assert html.index('localStorage.getItem("jbbind-theme")') < html.index("style.css")
    app = (ROOT / "jbbind/static/app.js").read_text()
    assert "document.documentElement.dataset.theme = choice" in app
    css = (ROOT / "jbbind/static/style.css").read_text()
    assert ':root[data-theme="dark"]' in css and ':root:not([data-theme="light"])' in css


# -------------------------------------------------------------- refinement

def _with_contacts(res: PredictionResult) -> PredictionResult:
    from dataclasses import replace
    from jbbind.core.refine import ContactGraph
    return replace(res, contacts=ContactGraph.from_edges([1, 2], [1], [2], [30.0]))


def test_refine_endpoint_rescores_without_rerunning_the_model(client):
    c, state, res = client
    state.results["refinejob"] = _with_contacts(res)
    r = c.post("/api/v1/jobs/refinejob/refine",
               json={"refine_method": "bayes", "refine_strength": 1.0})
    assert r.status_code == 200
    body = r.json()
    assert body["refinement"]["method"] == "bayes"
    low = body["residues"][1]
    # residue 2 (0.30) touches residue 1 (0.91), so its protein score rises
    assert low["pr"]["protein_nucleic"][0] > low["p"]["protein_nucleic"][0] == 0.30
    assert low["nb"] == [] and body["has_contacts"]      # nb is filled by predict

    # downloads follow the job's refinement
    csv = c.get("/api/v1/artifacts/refinejob/predictions.csv").text
    assert "protein_nucleic:Protein:refined" in csv.splitlines()[0]
    raw = c.get("/api/v1/artifacts/refinejob/predictions.pdb?setup=protein_nucleic").text
    ref = c.get("/api/v1/artifacts/refinejob/predictions.pdb"
                "?setup=protein_nucleic&source=refined").text
    assert raw != ref and "REMARK 100 refined" in ref
    for path in ("pymol.txt", "session.pml", "session.cxc", "figure.png"):
        assert c.get(f"/api/v1/artifacts/refinejob/{path}"
                     "?setup=protein_nucleic&source=refined").status_code == 200
    html = c.get("/api/v1/artifacts/refinejob/report.html?source=refined").text
    assert "Bayesian log-odds pooling" in html

    r = c.post("/api/v1/jobs/refinejob/refine", json={"refine_method": "gumbel"})
    assert r.json()["residues"][0]["psd"] is not None
    assert "refined_sd" in c.get("/api/v1/artifacts/refinejob/predictions.csv").text


def test_refine_endpoint_rejects_other_settings_and_bad_values(client):
    c, state, res = client
    state.results["refinejob"] = _with_contacts(res)
    r = c.post("/api/v1/jobs/refinejob/refine", json={"arch": "mlp"})
    assert r.json()["code"] == "InvalidSettings"
    r = c.post("/api/v1/jobs/refinejob/refine", json={"refine_strength": 3})
    assert r.json()["code"] == "InvalidSettings"
    assert c.post("/api/v1/jobs/nope/refine", json={}).status_code == 404


def test_refining_a_result_without_contacts_is_a_409(client):
    c, _, _ = client
    r = c.post("/api/v1/jobs/testjob/refine", json={"refine_method": "bayes"})
    assert r.status_code == 409 and r.json()["code"] == "RefinementUnavailable"


def test_refined_downloads_need_a_refinement(client):
    c, _, _ = client
    r = c.get("/api/v1/artifacts/testjob/predictions.pdb"
              "?setup=protein_nucleic&source=refined")
    assert r.status_code == 409
    assert c.get("/api/v1/artifacts/testjob/predictions.pdb"
                 "?setup=protein_nucleic&source=bogus").status_code == 409


def test_predict_page_offers_refinement_and_shows_the_change():
    html = (ROOT / "jbbind/static/index.html").read_text()
    for value in ("none", "bayes", "gumbel"):
        assert f'<option value="{value}"' in html
    assert 'id="score-source"' in html and '<option value="change">' in html
    js = (ROOT / "jbbind/static/predict.js").read_text()
    assert "/refine`" in js and "&source=${source}" in js
    css = (ROOT / "jbbind/static/style.css").read_text()
    assert css.count("--score-div-6:") == 3                 # light, dark, forced dark
