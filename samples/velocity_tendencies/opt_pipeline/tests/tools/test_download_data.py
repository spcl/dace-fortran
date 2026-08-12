"""Smoke tests for ``tools/download_data.sh``.

Exercises three modes without touching the real PolyBox URL:

  * ``LOCAL_DATA_DIR`` symlink mode (preferred for dev/CI -- avoids
    the 9.4 GB download)
  * idempotent skip on an already-populated OUTPUT_DIR
  * tarball fetch via ``URL`` (uses ``file://`` against a local tarball
    so no network access; only runs when ``curl`` is on PATH)

Each test runs the script in a clean temp dir so the production
``data_r02b05/`` is never touched.
"""

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO / "tools" / "download_data.sh"


def _run(env_extra: dict, cwd: Path | None = None):
    env = os.environ.copy()
    env.update(env_extra)
    return subprocess.run(
        ["bash", str(_SCRIPT)],
        env=env,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )


def test_local_data_dir_symlinks_instead_of_downloading(tmp_path: Path):
    """LOCAL_DATA_DIR=<path> -> OUTPUT_DIR becomes a symlink to <path>;
    no network access, no tarball touched."""
    src = tmp_path / "fake_local_data"
    src.mkdir()
    sentinel = src / "sentinel.bin"
    sentinel.write_text("nproma20480 sentinel payload")

    out = tmp_path / "data_r02b05"

    res = _run({"LOCAL_DATA_DIR": str(src), "OUTPUT_DIR": str(out)})
    assert res.returncode == 0, f"download_data.sh failed:\n{res.stderr}"

    assert out.is_symlink(), f"{out} should be a symlink"
    assert out.resolve() == src.resolve()
    # Sentinel reachable through the symlink.
    assert (out / "sentinel.bin").read_text() == "nproma20480 sentinel payload"


def test_local_data_dir_refuses_to_clobber_populated_output(tmp_path: Path):
    """If OUTPUT_DIR already has files, LOCAL_DATA_DIR mode bails
    instead of silently overwriting."""
    src = tmp_path / "fake_local_data"
    src.mkdir()
    (src / "from_local.bin").write_text("x")

    out = tmp_path / "data_r02b05"
    out.mkdir()
    (out / "preexisting.bin").write_text("DO NOT DELETE ME")

    res = _run({"LOCAL_DATA_DIR": str(src), "OUTPUT_DIR": str(out)})
    assert res.returncode != 0, "should refuse to clobber populated dir"
    assert "refusing to symlink over it" in res.stderr
    # Pre-existing file still there.
    assert (out / "preexisting.bin").read_text() == "DO NOT DELETE ME"
    assert not out.is_symlink()


def test_local_data_dir_replaces_empty_output_dir(tmp_path: Path):
    """An empty OUTPUT_DIR (e.g. left over from a prior mkdir -p) is
    safe to drop and replace with the symlink."""
    src = tmp_path / "fake_local_data"
    src.mkdir()
    (src / "datum.bin").write_text("y")

    out = tmp_path / "data_r02b05"
    out.mkdir()  # empty pre-existing dir

    res = _run({"LOCAL_DATA_DIR": str(src), "OUTPUT_DIR": str(out)})
    assert res.returncode == 0, res.stderr
    assert out.is_symlink()
    assert (out / "datum.bin").read_text() == "y"


def test_local_data_dir_replaces_stale_symlink(tmp_path: Path):
    """A second run with a different LOCAL_DATA_DIR re-points the
    symlink rather than failing on an existing one."""
    first = tmp_path / "first"
    first.mkdir()
    (first / "a.bin").write_text("first")

    second = tmp_path / "second"
    second.mkdir()
    (second / "b.bin").write_text("second")

    out = tmp_path / "data_r02b05"

    res = _run({"LOCAL_DATA_DIR": str(first), "OUTPUT_DIR": str(out)})
    assert res.returncode == 0, res.stderr
    assert out.resolve() == first.resolve()

    res = _run({"LOCAL_DATA_DIR": str(second), "OUTPUT_DIR": str(out)})
    assert res.returncode == 0, res.stderr
    assert out.resolve() == second.resolve()
    assert (out / "b.bin").read_text() == "second"


def test_idempotent_skip_when_output_dir_is_populated(tmp_path: Path):
    """Without LOCAL_DATA_DIR or URL change: a populated OUTPUT_DIR
    triggers an early exit (idempotent re-runs)."""
    out = tmp_path / "data_r02b05"
    out.mkdir()
    (out / "already_here.bin").write_text("z")

    # Use a deliberately-bogus URL so a *real* download attempt would
    # fail; this proves the script took the early-exit branch.
    res = _run({"URL": "http://localhost:9/should_never_be_fetched.tar.xz",
                "OUTPUT_DIR": str(out)})
    assert res.returncode == 0, f"expected idempotent skip; got:\n{res.stderr}"
    assert "already populated" in res.stdout
    assert (out / "already_here.bin").read_text() == "z"


def test_url_mode_via_local_file_tarball(tmp_path: Path):
    """Fetch + extract via ``file://`` URL -- exercises the curl/tar
    path without touching the network. Skips if ``curl`` is missing."""
    if shutil.which("curl") is None:
        pytest.skip("curl not on PATH")

    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    (payload_dir / "file_a.bin").write_text("aaa")
    (payload_dir / "file_b.bin").write_text("bbb")

    tarball = tmp_path / "fake_dataset.tar.xz"
    with tarfile.open(tarball, "w:xz") as tf:
        for f in sorted(payload_dir.iterdir()):
            tf.add(f, arcname=f.name)

    out = tmp_path / "data_r02b05"

    res = _run({"URL": f"file://{tarball}", "OUTPUT_DIR": str(out)})
    assert res.returncode == 0, f"download_data.sh failed:\n{res.stderr}"

    assert (out / "file_a.bin").read_text() == "aaa"
    assert (out / "file_b.bin").read_text() == "bbb"
    # Tarball cleaned up.
    assert not tarball.parent.joinpath("nproma20480_data_files.tar.xz").exists()
