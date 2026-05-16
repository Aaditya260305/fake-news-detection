"""Download GloVe 100d pretrained word embeddings (Wikipedia + Gigaword).

Why this script exists
----------------------
The original Stanford host (`nlp.stanford.edu`) is notoriously slow and
often times out half-way through the 822 MB zip. We therefore try a set
of mirrors in order, stream with a tqdm progress bar so you can see
what's happening, and support HTTP Range resume so a flaky connection
doesn't waste your previous progress.

Usage
-----
    python scripts/download_glove.py                  # default, tries all mirrors
    python scripts/download_glove.py --url <URL>      # force a single URL
    python scripts/download_glove.py --keep-zip       # keep the .zip after extract
    python scripts/download_glove.py --connect-timeout 10 --chunk 65536
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm


GLOVE_MIRRORS: tuple[str, ...] = (
    # HuggingFace CDN -- usually the fastest in India / Asia.
    "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
    # Stanford NLP (original, often slow / throttled).
    "https://nlp.stanford.edu/data/glove.6B.zip",
    "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
)

TARGET_FILE = "glove.6B.100d.txt"
OUT_DIR = Path("data/embeddings")


log = logging.getLogger("download_glove")


# ----------------------------------------------------------------------
#  download primitives
# ----------------------------------------------------------------------

def _human(bytes_: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(bytes_) < 1024:
            return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f} TB"


def _head_size(url: str, *, timeout: float) -> int | None:
    """Return ``Content-Length`` (bytes) for ``url`` or None on failure."""
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        if r.ok and "Content-Length" in r.headers:
            return int(r.headers["Content-Length"])
    except Exception as e:
        log.debug("HEAD %s failed: %s", url, e)
    return None


def _stream_download(
    url: str,
    out_path: Path,
    *,
    connect_timeout: float,
    chunk: int,
) -> bool:
    """Stream ``url`` to ``out_path`` with tqdm progress + Range resume.

    Returns True on success, False on any error (logged).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = out_path.with_suffix(out_path.suffix + ".part")
    existing = part_path.stat().st_size if part_path.exists() else 0

    headers = {}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        log.info("Resuming from byte %s", _human(existing))

    total = _head_size(url, timeout=connect_timeout)
    if total:
        log.info("Remote size: %s", _human(total))

    log.info("GET %s", url)
    t0 = time.time()
    try:
        # connect_timeout for the handshake; no read timeout -- a slow
        # mirror is allowed to take its time as long as bytes flow.
        with requests.get(
            url, stream=True, timeout=(connect_timeout, None), headers=headers
        ) as r:
            if r.status_code == 416:  # already complete
                log.info("Server says we already have the full file.")
                part_path.rename(out_path)
                return True
            r.raise_for_status()

            mode = "ab" if existing and r.status_code == 206 else "wb"
            if mode == "wb" and existing:
                log.warning("Server ignored Range header; restarting from 0.")
                existing = 0

            bar_total = total
            if bar_total and existing:
                bar_total = max(total, existing)

            with open(part_path, mode) as f:
                with tqdm(
                    total=bar_total,
                    initial=existing,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=out_path.name,
                    leave=True,
                ) as bar:
                    for c in r.iter_content(chunk_size=chunk):
                        if not c:
                            continue
                        f.write(c)
                        bar.update(len(c))
    except requests.exceptions.ConnectTimeout as e:
        log.error("Connect timeout: %s", e)
        return False
    except requests.exceptions.ChunkedEncodingError as e:
        log.error("Chunked encoding error (server cut us off): %s", e)
        return False
    except requests.exceptions.RequestException as e:
        log.error("Request failed: %s", e)
        return False
    except KeyboardInterrupt:
        log.warning("Interrupted by user. Partial file kept at %s for resume.", part_path)
        raise

    elapsed = time.time() - t0
    size = part_path.stat().st_size
    log.info(
        "Downloaded %s in %.1fs (avg %s/s)",
        _human(size),
        elapsed,
        _human(size / max(elapsed, 1e-6)),
    )
    part_path.replace(out_path)
    return True


def _try_mirrors(
    urls: Iterable[str], zip_path: Path, *, connect_timeout: float, chunk: int
) -> bool:
    for i, url in enumerate(urls, 1):
        log.info("--- Mirror %d/%d ---", i, len(list(urls) if not isinstance(urls, tuple) else urls))
        if _stream_download(url, zip_path, connect_timeout=connect_timeout, chunk=chunk):
            return True
        log.warning("Mirror failed, trying next one ...")
    return False


# ----------------------------------------------------------------------
#  main
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--url", default=None, help="force a specific download URL")
    p.add_argument("--connect-timeout", type=float, default=15.0)
    p.add_argument("--chunk", type=int, default=1 << 16, help="bytes per chunk (default 64K)")
    p.add_argument("--keep-zip", action="store_true", help="do not delete the zip after extract")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / TARGET_FILE
    if target.exists():
        log.info("GloVe already present at %s (%s) -- skipping download.", target, _human(target.stat().st_size))
        return 0

    urls = (args.url,) if args.url else GLOVE_MIRRORS
    zip_path = OUT_DIR / "glove.6B.zip"

    ok = _try_mirrors(urls, zip_path, connect_timeout=args.connect_timeout, chunk=args.chunk)
    if not ok:
        log.error("All mirrors failed. As a fallback you can download manually:")
        for u in GLOVE_MIRRORS:
            log.error("  %s", u)
        log.error("...then place the zip at %s and re-run this script.", zip_path)
        return 1

    log.info("Extracting %s from %s ...", TARGET_FILE, zip_path)
    try:
        with zipfile.ZipFile(zip_path) as z:
            members = {info.filename for info in z.infolist()}
            if TARGET_FILE not in members:
                log.error(
                    "Zip does not contain %s. Contents: %s",
                    TARGET_FILE,
                    ", ".join(sorted(members)),
                )
                return 2
            z.extract(TARGET_FILE, OUT_DIR)
    except zipfile.BadZipFile as e:
        log.error("Downloaded file is not a valid zip (%s). Deleting and aborting.", e)
        zip_path.unlink(missing_ok=True)
        return 3

    if not args.keep_zip:
        try:
            zip_path.unlink(missing_ok=True)
        except OSError as e:
            log.warning("Could not remove %s: %s", zip_path, e)

    log.info("Saved %s (%s)", target, _human(target.stat().st_size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
