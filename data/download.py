#!/usr/bin/env python
"""Download, extract and verify the real-world datasets of the SSRK reproduction package.

Two public datasets are used by the in-scope experiments (see ``data/README.md``):

* **PBMC 3k (processed AnnData)** -- ``data/pbmc/pbmc3k_processed.h5ad`` (52.6 MB).  The file is
  *shipped* with the package.  ``--pbmc`` only verifies it and re-downloads it from Zenodo
  (record 3886414, doi:10.5281/zenodo.3886414) when it is missing or corrupted.  Its license text
  (AGPL-3.0-only, ``data/pbmc/LICENSE``) and notice (``data/pbmc/NOTICE.md``) are shipped next to it;
  they are not data (this script leaves them alone; ``tests/test_data_checksums.py`` checks them).
* **UCI HAR** (Human Activity Recognition Using Smartphones, UCI ML Repository id 240,
  doi:10.24432/C54S4K).  The extracted dataset is 282.6 MB (269.5 MiB), so it is *not* shipped.
  ``--uci-har`` downloads the official archive, extracts the files read by the experiments into
  ``data/uci_har/UCI HAR Dataset/`` and verifies them.

Every file the experiments read is verified against the SHA-256 digests in ``data/checksums.json``
(they equal the digests of the files that produced the paper's numbers).  The experiments only
ever see files whose digest matched; a mismatch is a hard error.

Properties of the downloader
----------------------------
* Standard library only (``urllib``, ``hashlib``, ``zipfile``); honours ``HTTP(S)_PROXY``.
* Resume-safe: data are streamed into ``<name>.part`` files, interrupted downloads are resumed
  with HTTP ``Range`` requests, and a file is moved into place (``os.replace``) only after its
  size and SHA-256 have been checked.  A crash never leaves a truncated file under a final name.
* Retries with exponential back-off (``Retry-After`` honoured for HTTP 429/503), mirror URLs.
* Idempotent: files that are already present and verified are skipped (no network access).
* Archive digests are recorded for information.  If the UCI repository ever re-packages the
  archive (different archive digest), extraction proceeds and the *extracted files* are verified
  individually; only their digests decide success.

Usage (from the package root)::

    python data/download.py --all            # PBMC check (+ repair) and UCI HAR download
    python data/download.py --uci-har        # UCI HAR only
    python data/download.py --pbmc           # verify / repair the shipped PBMC file
    python data/download.py --check          # verify only, never touches the network
    python data/download.py --uci-har --uci-full   # also extract the raw 'Inertial Signals'

Exit status: 0 if every requested file is present and verified, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional


def _shown(path) -> str:
    """``path`` relative to the package root when it lies inside it (keeps logs free of machine-specific prefixes)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        return str(p)

DATA_DIR = Path(__file__).resolve().parent          # PKG/data
CHECKSUMS_PATH = DATA_DIR / "checksums.json"
CHUNK = 1 << 20                                       # 1 MiB read/hash block
USER_AGENT = "SSRK-reproduction-data-downloader/1.0 (python-urllib)"
RETRYABLE_HTTP = {408, 425, 429, 500, 502, 503, 504}


class DataError(RuntimeError):
    """Raised when a dataset cannot be obtained or fails verification."""


# --------------------------------------------------------------------------------------------
# Manifest and verification
# --------------------------------------------------------------------------------------------
def load_checksums(path: Path = CHECKSUMS_PATH) -> dict:
    """Return the parsed ``checksums.json`` manifest."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path: Path, chunk: int = CHUNK) -> str:
    """SHA-256 hex digest of a file, streamed in ``chunk``-byte blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, spec: dict) -> tuple[bool, str]:
    """Check ``path`` against a manifest entry ``{"bytes": int, "sha256": str}``.

    Returns ``(ok, reason)``; the size is checked first so that a truncated file is reported
    without hashing it.
    """
    path = Path(path)
    if not path.is_file():
        return False, "missing"
    size = path.stat().st_size
    if "bytes" in spec and size != int(spec["bytes"]):
        return False, f"size {size} != expected {spec['bytes']}"
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        return False, f"sha256 {digest} != expected {spec['sha256']}"
    return True, "ok"


def dataset_files(manifest: dict, name: str, include_optional: bool = False) -> dict[str, dict]:
    """Manifest entries (relative path -> spec) of dataset ``name``.

    Entries with ``"required": false`` (e.g. the raw UCI HAR inertial signals, which no
    experiment reads) are included only when ``include_optional`` is true.
    """
    files = manifest["datasets"][name]["files"]
    return {rel: spec for rel, spec in files.items()
            if include_optional or spec.get("required", True)}


def verify_dataset(data_dir: Path, manifest: dict, name: str,
                   include_optional: bool = False, quiet: bool = False) -> dict[str, str]:
    """Verify every file of dataset ``name`` under ``data_dir``.

    Returns a dict ``relative path -> reason`` for the files that failed (empty if all passed).
    """
    failures: dict[str, str] = {}
    for rel, spec in dataset_files(manifest, name, include_optional).items():
        ok, reason = verify_file(Path(data_dir) / rel, spec)
        if not ok:
            failures[rel] = reason
        if not quiet:
            tag = "OK" if ok else ("MISSING" if reason == "missing" else "FAIL")
            print(f"  [{tag}] {rel}" + ("" if ok or reason == "missing" else f"  ({reason})"))
    return failures


# --------------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------------
def _fmt_mb(n: float) -> str:
    return f"{n / 1e6:7.1f} MB"


class _Progress:
    """Minimal progress reporter (carriage-return line on a TTY, 10 % steps otherwise)."""

    def __init__(self, label: str, total: Optional[int], start: int, quiet: bool):
        self.label, self.total, self.quiet = label, total, quiet
        self.start_bytes, self.t0 = start, time.monotonic()
        self.tty = sys.stderr.isatty()
        self.last_t, self.last_decile, self.last_done = 0.0, -1, -1

    def update(self, done: int, final: bool = False) -> None:
        if self.quiet:
            return
        now = time.monotonic()
        if self.tty:
            show = final or now - self.last_t >= 0.5
        else:
            decile = int(10 * done / self.total) if self.total else -1
            show = (final and done != self.last_done) or decile > self.last_decile
            self.last_decile = max(self.last_decile, decile)
        if not show:
            return
        rate = (done - self.start_bytes) / max(now - self.t0, 1e-6)
        pct = f"{100.0 * done / self.total:5.1f}%" if self.total else "   ?  "
        tot = _fmt_mb(self.total) if self.total else "      ?   "
        line = f"  {self.label}: {_fmt_mb(done)} / {tot} {pct}  {rate / 1e6:6.2f} MB/s"
        if self.tty:
            sys.stderr.write("\r" + line + ("\n" if final else ""))
        else:
            sys.stderr.write(line + "\n")
        sys.stderr.flush()
        self.last_t, self.last_done = now, done


def _content_total(resp, offset: int) -> Optional[int]:
    """Total size of the remote file from Content-Range (206) or Content-Length (200)."""
    crange = resp.headers.get("Content-Range")
    if crange and "/" in crange:
        tail = crange.rsplit("/", 1)[1].strip()
        if tail.isdigit():
            return int(tail)
    length = resp.headers.get("Content-Length")
    if length and length.isdigit():
        return int(length) + offset
    return None


def _stream_once(url: str, part: Path, timeout: float, label: str,
                 quiet: bool) -> tuple[Optional[int], int]:
    """One HTTP attempt: append to ``part``, resuming from its current size.

    Returns ``(total, offset)``: the total size announced by the server (``None`` if unknown) and
    the byte offset the transfer actually resumed from (0 if the server ignored the ``Range``
    header and the file was restarted).  Raises on network errors; the caller retries.
    """
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset:           # range not satisfiable: .part already complete?
            return offset, offset
        raise
    with resp:
        status = getattr(resp, "status", None) or resp.getcode()
        if offset and status != 206:             # server ignored the Range header: start over
            offset = 0
        total = _content_total(resp, offset)
        mode = "ab" if offset else "wb"
        progress = _Progress(label, total, offset, quiet)
        done = offset
        with open(part, mode) as fh:
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                progress.update(done)
        progress.update(done, final=True)
    return total, offset


def download(urls: Iterable[str], dest: Path, expected_sha256: Optional[str] = None,
             expected_bytes: Optional[int] = None, strict: bool = True, retries: int = 5,
             timeout: float = 60.0, quiet: bool = False) -> str:
    """Download the first working URL of ``urls`` to ``dest`` (atomic, resumable).

    The data are written to ``dest + '.part'``; an existing ``.part`` file is resumed.  After the
    transfer the file size is compared with the size announced by the server, and the SHA-256 with
    ``expected_sha256``.  With ``strict=True`` a digest mismatch is an error (the partial file is
    deleted); with ``strict=False`` a mismatch only prints a warning (used for archives whose
    *contents* are verified afterwards).  A resumed transfer whose digest does not match is
    retried once from scratch before giving up, in case the remote file changed in between.

    Returns the SHA-256 of the downloaded file.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    errors: list[str] = []
    for url in urls:
        fresh_retry_done = False
        attempt = 0
        last_error = "no response"
        while attempt <= retries:
            have = part.stat().st_size if part.exists() else 0
            if expected_bytes is not None and have > int(expected_bytes):
                part.unlink()                     # stale/oversized partial file
                have = 0
            try:
                if not quiet:
                    action = f"resuming at {_fmt_mb(have).strip()}" if have else "downloading"
                    print(f"  {action}: {url}")
                total, resumed_from = _stream_once(url, part, timeout, dest.name, quiet)
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRYABLE_HTTP:
                    errors.append(f"{url}: HTTP {exc.code} {exc.reason}")
                    break                         # permanent error: try the next mirror
                last_error = f"HTTP {exc.code} {exc.reason}"
                wait = exc.headers.get("Retry-After") if exc.headers else None
                delay = float(wait) if wait and wait.isdigit() else min(2.0 ** attempt, 60.0)
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError,
                    TimeoutError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                delay = min(2.0 ** attempt, 60.0)
            else:
                delay = None
            if delay is not None:                 # retryable failure of this attempt
                attempt += 1
                if attempt <= retries:
                    print(f"  {last_error}; retry {attempt}/{retries} in {delay:.0f} s",
                          file=sys.stderr)
                    time.sleep(delay)
                continue

            size = part.stat().st_size
            if total is not None and size < total:     # connection dropped: resume
                last_error = f"transfer ended at {size} of {total} bytes"
                attempt += 1
                continue
            digest = sha256_file(part)
            if expected_sha256 and digest != expected_sha256:
                if resumed_from and not fresh_retry_done:  # resumed onto a different remote file?
                    print("  digest mismatch after a resumed transfer; restarting from scratch",
                          file=sys.stderr)
                    part.unlink()
                    fresh_retry_done = True
                    continue
                if strict:
                    part.unlink()
                    raise DataError(
                        f"SHA-256 mismatch for {dest.name} from {url}:\n"
                        f"  got      {digest}\n  expected {expected_sha256}\n"
                        "The remote file differs from the file used for the paper.")
                print(f"  WARNING: {dest.name} has SHA-256 {digest}, recorded value is "
                      f"{expected_sha256}. The archive was probably re-packaged upstream; its "
                      "contents are verified file by file next.", file=sys.stderr)
            os.replace(part, dest)
            return digest
        else:
            errors.append(f"{url}: gave up after {retries + 1} attempts (last error: {last_error})")
    raise DataError("Download failed for " + dest.name + ":\n  " + "\n  ".join(errors) +
                    "\nCheck your network/proxy settings, or download the file by hand (see "
                    "data/README.md).")


# --------------------------------------------------------------------------------------------
# Extraction (UCI HAR)
# --------------------------------------------------------------------------------------------
def _members(zf: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    return {info.filename.replace("\\", "/"): info for info in zf.infolist() if not info.is_dir()}


def _extract_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, dest: Path,
                    spec: Optional[dict]) -> str:
    """Stream one zip member to ``dest`` via a ``.part`` file; verify against ``spec`` if given.

    The destination path is built from the manifest, never from the archive's member names, so a
    malformed archive cannot write outside the data directory.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    with zf.open(info) as src, open(part, "wb") as out:
        for block in iter(lambda: src.read(CHUNK), b""):
            digest.update(block)
            out.write(block)
    hexd = digest.hexdigest()
    if spec is not None and hexd != spec["sha256"]:
        part.unlink()
        raise DataError(f"{dest}: extracted file has SHA-256 {hexd}, expected {spec['sha256']}.")
    os.replace(part, dest)
    return hexd


def extract_uci_har(zip_path: Path, data_dir: Path, manifest: dict, work_dir: Path,
                    include_optional: bool = False, force: bool = False,
                    quiet: bool = False) -> None:
    """Extract the UCI HAR files listed in the manifest from ``zip_path`` into ``data_dir``.

    ``zip_path`` may be either the archive served by the UCI repository (which wraps a second
    archive ``UCI HAR Dataset.zip``) or that inner archive itself.  Files that are already
    present and verified are left untouched unless ``force`` is set.
    """
    ds = manifest["datasets"]["uci_har"]
    inner_name = ds["inner_archive"]["name"]
    root = ds["extract_root"]                                 # "uci_har"
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise DataError(f"Archive not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise DataError(f"{zip_path} is not a zip archive (incomplete download?).")

    with zipfile.ZipFile(zip_path) as zf:
        names = _members(zf)
        if inner_name in names:                               # outer UCI archive
            inner_path = Path(work_dir) / inner_name
            spec = ds["inner_archive"]
            ok, _ = verify_file(inner_path, spec) if inner_path.exists() else (False, "")
            if not ok:
                if not quiet:
                    print(f"  unpacking inner archive '{inner_name}'")
                hexd = _extract_member(zf, names[inner_name], inner_path, None)
                if hexd != spec["sha256"] and not quiet:
                    print(f"  WARNING: inner archive SHA-256 {hexd} differs from the recorded "
                          f"{spec['sha256']}; verifying the extracted files individually.",
                          file=sys.stderr)
            extract_uci_har(inner_path, data_dir, manifest, work_dir, include_optional, force, quiet)
            return
        if not any(n.startswith("UCI HAR Dataset/") for n in names):
            raise DataError(f"{zip_path} does not look like the UCI HAR archive (no "
                            f"'{inner_name}' and no 'UCI HAR Dataset/' entries).")
        for rel, spec in dataset_files(manifest, "uci_har", include_optional).items():
            member = Path(rel).relative_to(root).as_posix()   # e.g. "UCI HAR Dataset/train/X_train.txt"
            dest = Path(data_dir) / rel
            if not force and verify_file(dest, spec)[0]:
                if not quiet:
                    print(f"  [skip] {rel} (already verified)")
                continue
            if member not in names:
                raise DataError(f"{zip_path} has no member '{member}'.")
            _extract_member(zf, names[member], dest, spec)
            if not quiet:
                print(f"  [extracted] {rel}")


# --------------------------------------------------------------------------------------------
# Dataset front-ends
# --------------------------------------------------------------------------------------------
def fetch_pbmc(data_dir: Path, manifest: dict, retries: int, timeout: float,
               offline: bool = False, quiet: bool = False) -> bool:
    """Verify the shipped PBMC 3k file; re-download it from Zenodo if missing or corrupted."""
    print("PBMC 3k (processed AnnData, Zenodo 3886414)")
    ok = True
    for rel, spec in dataset_files(manifest, "pbmc").items():
        dest = Path(data_dir) / rel
        good, reason = verify_file(dest, spec)
        if good:
            print(f"  [OK] {rel}")
            continue
        print(f"  [{'MISSING' if reason == 'missing' else 'CORRUPT'}] {rel} ({reason})")
        if offline:
            ok = False
            continue
        download(spec["urls"], dest, expected_sha256=spec["sha256"], expected_bytes=spec["bytes"],
                 strict=True, retries=retries, timeout=timeout, quiet=quiet)
        good, reason = verify_file(dest, spec)
        print(f"  [{'OK' if good else 'FAIL'}] {rel}" + ("" if good else f" ({reason})"))
        ok &= good
    return ok


def fetch_uci_har(data_dir: Path, manifest: dict, archive_dir: Path, retries: int, timeout: float,
                  include_optional: bool = False, force: bool = False, offline: bool = False,
                  quiet: bool = False) -> bool:
    """Download (if needed), extract and verify UCI HAR into ``data_dir/uci_har``."""
    print("UCI HAR (UCI Machine Learning Repository, id 240)")
    failures = verify_dataset(data_dir, manifest, "uci_har", include_optional, quiet=quiet)
    if not failures and not force:
        print("  all files present and verified; nothing to do")
        return True
    if offline:
        return False
    ds = manifest["datasets"]["uci_har"]
    arch = ds["archive"]
    archive_dir = Path(archive_dir)
    zip_path = archive_dir / arch["name"]
    have_archive = zip_path.is_file() and verify_file(zip_path, arch)[0]
    if have_archive:
        print(f"  using cached archive {_shown(zip_path)}")
    else:
        # The UCI server streams the archive without Content-Length and ignores Range requests,
        # so completeness is established by the zip structure and the per-file digests.
        download(arch["urls"], zip_path, expected_sha256=arch["sha256"], expected_bytes=None,
                 strict=False, retries=retries, timeout=timeout, quiet=quiet)
        if not zipfile.is_zipfile(zip_path):
            zip_path.unlink()
            raise DataError(f"{zip_path.name} is not a valid zip archive (truncated transfer?); "
                            "it was deleted. Please run the command again.")
    extract_uci_har(zip_path, data_dir, manifest, archive_dir, include_optional, force, quiet)
    failures = verify_dataset(data_dir, manifest, "uci_har", include_optional, quiet=quiet)
    return not failures


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download, extract and verify the datasets of the SSRK reproduction package.")
    sel = parser.add_argument_group("datasets (default: --all)")
    sel.add_argument("--all", action="store_true", help="PBMC 3k and UCI HAR")
    sel.add_argument("--pbmc", action="store_true",
                     help="verify the shipped PBMC file; re-download it if missing or corrupted")
    sel.add_argument("--uci-har", action="store_true", help="download/extract/verify UCI HAR")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="data root (default: the package's data/ directory)")
    parser.add_argument("--archive-dir", type=Path, default=None,
                        help="where downloaded archives are kept (default: <data-dir>/_downloads)")
    parser.add_argument("--check", action="store_true",
                        help="verify only; never access the network")
    parser.add_argument("--uci-full", action="store_true",
                        help="also extract and verify the raw 'Inertial Signals' (not used)")
    parser.add_argument("--force", action="store_true",
                        help="re-extract UCI HAR files even if they already verify")
    parser.add_argument("--retries", type=int, default=5, help="retries per URL (default 5)")
    parser.add_argument("--timeout", type=float, default=60.0, help="socket timeout in s")
    parser.add_argument("--quiet", action="store_true", help="less output")
    args = parser.parse_args(argv)
    try:                                   # keep stdout and the stderr progress lines in order
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    if not (args.all or args.pbmc or args.uci_har):
        args.all = True
    data_dir = args.data_dir.resolve()
    archive_dir = (args.archive_dir or data_dir / "_downloads").resolve()
    manifest = load_checksums()
    print(f"data directory: {_shown(data_dir)}")
    ok = True
    try:
        if args.all or args.pbmc:
            ok &= fetch_pbmc(data_dir, manifest, args.retries, args.timeout, args.check, args.quiet)
        if args.all or args.uci_har:
            ok &= fetch_uci_har(data_dir, manifest, archive_dir, args.retries, args.timeout,
                                args.uci_full, args.force, args.check, args.quiet)
    except DataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; partial downloads are kept as *.part and resumed next time",
              file=sys.stderr)
        return 1
    print("all requested datasets verified" if ok else
          "verification FAILED (see above); run without --check to download/repair")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
