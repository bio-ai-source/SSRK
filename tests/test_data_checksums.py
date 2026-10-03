"""Tests of the data layer (data/download.py, data/prepare_uci_har.py, data/checksums.json).

* The manifest is well formed (64-hex SHA-256 digests, byte sizes, URLs, required flags) and
  equals the original study's record ``reference/data/dataset_hashes_release.json``.
* The shipped PBMC 3k file ``data/pbmc/pbmc3k_processed.h5ad`` has the recorded size, SHA-256 and
  the MD5 published on the Zenodo record (hashed here with ``hashlib``, independently of
  ``download.py``).  Its license text ``data/pbmc/LICENSE`` is the unmodified canonical AGPL-3.0
  text, and ``data/pbmc/NOTICE.md`` names the record, DOI, license and SHA-256 of the file.
* If UCI HAR has been downloaded, every extracted file matches its digest; if the optional array
  cache exists it is bitwise equal to the text files.  (Skipped when the data are absent.)  These
  checks look in ``PKG/data`` unless the environment variable ``SSRK_DATA_DIR`` names another
  data root (the layout below it must be the same).
* Offline tests of the machinery: nested-zip extraction with verification and idempotence,
  rejection of a corrupted member, the bitwise cache check, and resumable / restarting /
  retrying / mirror-falling-back / strict downloads against a local HTTP server on 127.0.0.1.
"""
from __future__ import annotations

import hashlib
import http.server
import io
import json
import os
import re
import sys
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
DATA = PKG / "data"
sys.path.insert(0, str(DATA))

import download as dl  # noqa: E402
import prepare_uci_har as prep  # noqa: E402

MANIFEST = json.loads((DATA / "checksums.json").read_text(encoding="utf-8"))
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _hashes(path: Path) -> tuple[int, str, str]:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            sha.update(block)
            md5.update(block)
    return path.stat().st_size, sha.hexdigest(), md5.hexdigest()


# --------------------------------------------------------------------------------- manifest
def test_manifest_schema():
    assert set(MANIFEST["datasets"]) == {"pbmc", "uci_har"}
    for name, ds in MANIFEST["datasets"].items():
        assert ds["files"], name
        for rel, spec in ds["files"].items():
            assert not Path(rel).is_absolute() and ".." not in Path(rel).parts, rel
            assert HEX64.match(spec["sha256"]), rel
            assert isinstance(spec["bytes"], int) and spec["bytes"] > 0, rel
            assert isinstance(spec.get("required", True), bool), rel
    pbmc = MANIFEST["datasets"]["pbmc"]["files"]["pbmc/pbmc3k_processed.h5ad"]
    assert pbmc["bytes"] == 52633500
    assert all(u.startswith("https://zenodo.org/") for u in pbmc["urls"])
    uci = MANIFEST["datasets"]["uci_har"]
    assert uci["archive"]["urls"][0].startswith("https://archive.ics.uci.edu/static/public/240/")
    required = {Path(r).relative_to("uci_har/UCI HAR Dataset").as_posix()
                for r, s in uci["files"].items() if s.get("required", True)}
    assert {"features.txt", "activity_labels.txt", "train/X_train.txt", "train/y_train.txt",
            "train/subject_train.txt", "test/X_test.txt", "test/y_test.txt",
            "test/subject_test.txt"} <= required
    assert len(uci["files"]) == 28                       # 10 required + 18 inertial signals


def test_manifest_equals_original_study_record():
    """Every digest equals the record written by the original study (reference/data copy)."""
    record = json.loads((PKG / "reference" / "data" / "dataset_hashes_release.json")
                        .read_text(encoding="utf-8"))
    for ds in MANIFEST["datasets"].values():
        for rel, spec in ds["files"].items():
            orig = record["data/" + rel]
            assert (spec["bytes"], spec["sha256"]) == (orig["bytes"], orig["sha256"]), rel


# --------------------------------------------------------------------------------- shipped data
def test_shipped_pbmc_file_hash():
    spec = MANIFEST["datasets"]["pbmc"]["files"]["pbmc/pbmc3k_processed.h5ad"]
    path = DATA / "pbmc" / "pbmc3k_processed.h5ad"
    assert path.is_file(), "shipped PBMC file missing; run `python data/download.py --pbmc`"
    size, sha, md5 = _hashes(path)
    assert size == spec["bytes"]
    assert sha == spec["sha256"] == "52f5becad785656ab391d2b41a91bf71344a1f0a39e19d510e61668d6038ad0f"
    assert md5 == spec["md5"]                            # MD5 published on Zenodo record 3886414


AGPL3_SHA256 = "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"  # gnu.org/licenses/agpl-3.0.txt


def test_shipped_pbmc_license_and_notice():
    """License text and notice that accompany the redistributed AGPL-3.0-only PBMC file."""
    pbmc = MANIFEST["datasets"]["pbmc"]
    spec = pbmc["license_text"]
    assert spec["path"] not in pbmc["files"]             # not a data file: download.py never fetches it
    path = DATA / spec["path"]
    assert path.is_file(), "data/pbmc/LICENSE (AGPL-3.0 text) missing"
    size, sha, _ = _hashes(path)
    assert (size, sha) == (spec["bytes"], spec["sha256"]) == (34523, AGPL3_SHA256)
    assert path.read_bytes().lstrip().startswith(b"GNU AFFERO GENERAL PUBLIC LICENSE")
    notice = (DATA / pbmc["notice"]).read_text(encoding="utf-8")
    data_sha = pbmc["files"]["pbmc/pbmc3k_processed.h5ad"]["sha256"]
    for needle in (data_sha, pbmc["doi"], "Zenodo record 3886414", "AGPL-3.0-only", "unmodified",
                   "Klas Hatje", "Zheng et al., 2017", "`LICENSE`"):
        assert needle in notice, needle


DATA_ROOT = Path(os.environ.get("SSRK_DATA_DIR", str(DATA)))
UCI_FILES = sorted(MANIFEST["datasets"]["uci_har"]["files"].items())
UCI_DIR = DATA_ROOT / "uci_har" / "UCI HAR Dataset"


@pytest.mark.parametrize("rel,spec", UCI_FILES, ids=[r.split("Dataset/")[1] for r, _ in UCI_FILES])
def test_uci_har_files_if_present(rel, spec):
    if not UCI_DIR.is_dir():
        pytest.skip("UCI HAR not downloaded (run `python data/download.py --uci-har`)")
    path = DATA_ROOT / rel
    if not path.exists() and not spec.get("required", True):
        pytest.skip("optional file not extracted (use --uci-full)")
    assert path.is_file(), f"{rel} missing; run `python data/download.py --uci-har`"
    size, sha, _ = _hashes(path)
    assert (size, sha) == (spec["bytes"], spec["sha256"])


def test_uci_cache_if_present():
    cache = DATA_ROOT / "uci_har" / prep.CACHE_NAME
    if not (cache.is_file() and UCI_DIR.is_dir()):
        pytest.skip("no UCI HAR array cache (python data/prepare_uci_har.py --cache)")
    arrays = prep.load_uci_har_text(UCI_DIR)
    prep.check_shapes(arrays)
    assert prep.verify_cache(cache, arrays) == []


def test_cache_roundtrip_detects_differences(tmp_path):
    rng = np.random.default_rng(0)
    arrays = {"X_train": rng.standard_normal((5, 3)).astype(np.float32),
              "y_train": np.arange(5) - 1,
              "feature_names": np.array(["tA", "fB", "angle(C)"])}
    cache = tmp_path / "cache.npz"
    prep.write_cache(arrays, cache, {"a.txt": "0" * 64})
    assert prep.verify_cache(cache, arrays, {"a.txt": "0" * 64}) == []
    changed = dict(arrays, X_train=arrays["X_train"].copy())
    changed["X_train"][0, 0] = np.nextafter(changed["X_train"][0, 0], np.float32(np.inf))
    assert prep.verify_cache(cache, changed) == ["X_train"]          # one-ulp change detected
    assert prep.verify_cache(cache, dict(arrays, y_train=arrays["y_train"].astype(np.int32))) == ["y_train"]
    assert prep.verify_cache(cache, arrays, {"a.txt": "1" * 64}) == ["meta_json.source_sha256"]
    assert prep.bitwise_equal(np.float32(0.0).reshape(1), np.float32(-0.0).reshape(1)) is False


# --------------------------------------------------------------------------------- verification
def test_verify_file_detects_truncation_and_corruption(tmp_path):
    payload = b"0123456789" * 1000
    path = tmp_path / "f.bin"
    path.write_bytes(payload)
    spec = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    assert dl.verify_file(path, spec) == (True, "ok")
    assert dl.verify_file(tmp_path / "absent.bin", spec) == (False, "missing")
    path.write_bytes(payload[:-1])
    assert dl.verify_file(path, spec)[0] is False
    path.write_bytes(payload[:-1] + b"X")
    ok, reason = dl.verify_file(path, spec)
    assert not ok and reason.startswith("sha256")


# --------------------------------------------------------------------------------- extraction
def _fake_uci(tmp_path: Path, corrupt: str | None = None):
    """Nested zip mimicking the UCI archive plus a manifest describing its members."""
    members = {
        "UCI HAR Dataset/features.txt": b"1 tBodyAcc-mean()-X\n2 fBodyAcc-mean()-X\n",
        "UCI HAR Dataset/train/X_train.txt": b" 1.0e-01 -2.5e-01\n 3.0e-01  4.0e-01\n",
        "UCI HAR Dataset/test/X_test.txt": b" 5.0e-01  6.0e-01\n",
    }
    files = {"uci_har/" + m: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(),
                              "required": True} for m, b in members.items()}
    inner_buf = io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w", zipfile.ZIP_DEFLATED) as z:
        for m, b in members.items():
            z.writestr(m, b + (b"tampered" if m == corrupt else b""))
        z.writestr("__MACOSX/UCI HAR Dataset/._features.txt", b"junk")
    inner = inner_buf.getvalue()
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("UCI HAR Dataset.names", b"readme")
        z.writestr("UCI HAR Dataset.zip", inner)
    manifest = {"datasets": {"uci_har": {
        "extract_root": "uci_har", "files": files,
        "inner_archive": {"name": "UCI HAR Dataset.zip", "bytes": len(inner),
                          "sha256": hashlib.sha256(inner).hexdigest()}}}}
    return outer, manifest


def test_extract_nested_zip_verifies_and_is_idempotent(tmp_path, capsys):
    outer, manifest = _fake_uci(tmp_path)
    data_dir, work = tmp_path / "data", tmp_path / "work"
    dl.extract_uci_har(outer, data_dir, manifest, work)
    assert dl.verify_dataset(data_dir, manifest, "uci_har", quiet=True) == {}
    assert not (data_dir / "uci_har" / "__MACOSX").exists()       # only listed files extracted
    assert not list(data_dir.rglob("*.part"))
    target = data_dir / "uci_har" / "UCI HAR Dataset" / "train" / "X_train.txt"
    mtime = target.stat().st_mtime_ns
    capsys.readouterr()
    dl.extract_uci_har(outer, data_dir, manifest, work)            # second run: all skipped
    assert target.stat().st_mtime_ns == mtime
    assert capsys.readouterr().out.count("[skip]") == 3
    # the inner archive alone is accepted as well (manual download of "UCI HAR Dataset.zip")
    dl.extract_uci_har(work / "UCI HAR Dataset.zip", tmp_path / "data2", manifest, work)
    assert dl.verify_dataset(tmp_path / "data2", manifest, "uci_har", quiet=True) == {}


def test_extract_rejects_corrupted_member(tmp_path):
    outer, manifest = _fake_uci(tmp_path, corrupt="UCI HAR Dataset/train/X_train.txt")
    data_dir = tmp_path / "data"
    with pytest.raises(dl.DataError, match="SHA-256"):
        dl.extract_uci_har(outer, data_dir, manifest, tmp_path / "work")
    bad = data_dir / "uci_har" / "UCI HAR Dataset" / "train" / "X_train.txt"
    assert not bad.exists() and not bad.with_name(bad.name + ".part").exists()


def test_extract_rejects_foreign_archive(tmp_path):
    _, manifest = _fake_uci(tmp_path)
    other = tmp_path / "other.zip"
    with zipfile.ZipFile(other, "w") as z:
        z.writestr("something/else.txt", b"x")
    with pytest.raises(dl.DataError, match="does not look like"):
        dl.extract_uci_har(other, tmp_path / "data", manifest, tmp_path / "work")


# --------------------------------------------------------------------------------- downloads
class _Server:
    """Tiny HTTP server on 127.0.0.1 serving one payload, optionally honouring Range requests."""

    def __init__(self, payload: bytes, honour_range: bool = True, fail_first: int = 0):
        outer = self
        self.payload, self.honour_range, self.ranges = payload, honour_range, []
        self.fail_first = fail_first

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path != "/file.bin":
                    self.send_error(404)
                    return
                rng = self.headers.get("Range")
                outer.ranges.append(rng)
                if outer.fail_first > 0:                         # transient server error
                    outer.fail_first -= 1
                    self.send_response(503)
                    self.send_header("Retry-After", "0")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = outer.payload
                if rng and outer.honour_range:
                    start = int(rng.split("=", 1)[1].split("-", 1)[0])
                    body = data[start:]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    body = data
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(autouse=True)
def _no_proxy_for_localhost(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


PAYLOAD = bytes(range(256)) * 4096                                  # 1 MiB
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


def test_download_resumes_partial_file(tmp_path):
    dest = tmp_path / "file.bin"
    dest.with_name("file.bin.part").write_bytes(PAYLOAD[:300_000])  # interrupted earlier run
    with _Server(PAYLOAD) as srv:
        digest = dl.download([srv.base + "/file.bin"], dest, PAYLOAD_SHA, len(PAYLOAD),
                             retries=1, timeout=10, quiet=True)
    assert srv.ranges == ["bytes=300000-"]
    assert digest == PAYLOAD_SHA and dest.read_bytes() == PAYLOAD
    assert not dest.with_name("file.bin.part").exists()


def test_download_restarts_when_range_ignored_and_uses_mirror(tmp_path):
    dest = tmp_path / "file.bin"
    dest.with_name("file.bin.part").write_bytes(b"garbage" * 1000)
    with _Server(PAYLOAD, honour_range=False) as srv:
        digest = dl.download([srv.base + "/missing", srv.base + "/file.bin"], dest, PAYLOAD_SHA,
                             retries=1, timeout=10, quiet=True)
    assert digest == PAYLOAD_SHA and dest.read_bytes() == PAYLOAD


def test_download_restarts_after_resume_onto_changed_file(tmp_path):
    dest = tmp_path / "file.bin"
    dest.with_name("file.bin.part").write_bytes(b"#" * 300_000)  # prefix of another file
    with _Server(PAYLOAD) as srv:
        digest = dl.download([srv.base + "/file.bin"], dest, PAYLOAD_SHA, len(PAYLOAD),
                             retries=1, timeout=10, quiet=True)
    assert srv.ranges == ["bytes=300000-", None]                     # resumed, then restarted
    assert digest == PAYLOAD_SHA and dest.read_bytes() == PAYLOAD


def test_download_retries_transient_errors(tmp_path):
    dest = tmp_path / "file.bin"
    with _Server(PAYLOAD, fail_first=2) as srv:
        digest = dl.download([srv.base + "/file.bin"], dest, PAYLOAD_SHA, retries=2, timeout=10,
                             quiet=True)
    assert len(srv.ranges) == 3 and digest == PAYLOAD_SHA
    with _Server(PAYLOAD, fail_first=5) as srv:
        with pytest.raises(dl.DataError, match="gave up after 2 attempts"):
            dl.download([srv.base + "/file.bin"], tmp_path / "g.bin", PAYLOAD_SHA, retries=1,
                        timeout=10, quiet=True)


def test_download_strict_mismatch_leaves_nothing(tmp_path):
    dest = tmp_path / "file.bin"
    with _Server(PAYLOAD) as srv:
        with pytest.raises(dl.DataError, match="SHA-256 mismatch"):
            dl.download([srv.base + "/file.bin"], dest, "0" * 64, retries=1, timeout=10,
                        quiet=True)
    assert not dest.exists() and not dest.with_name("file.bin.part").exists()


def test_check_mode_is_offline(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network access in --check mode")
    monkeypatch.setattr(dl.urllib.request, "urlopen", no_network)
    assert dl.main(["--check", "--uci-har", "--data-dir", str(tmp_path), "--quiet"]) == 1
    assert dl.main(["--check", "--pbmc", "--quiet"]) == 0          # shipped PBMC file verifies
