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
