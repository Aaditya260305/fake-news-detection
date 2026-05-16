"""Download the two Kaggle datasets used in the EKNet paper (English only).

Datasets
--------
1. Real or Fake (rchitic17/real-or-fake)            -> primary corpus
2. Fake News Detection (jruvika/fake-news-detection) -> held-out test

Authentication
--------------
The Kaggle CLI accepts any of:
    1) %USERPROFILE%\\.kaggle\\kaggle.json
       (JSON: {"username": "...", "key": "..."})
    2) %USERPROFILE%\\.kaggle\\access_token
       (single-line KGAT_xxxxxxxx token)
    3) Env vars: KAGGLE_USERNAME + KAGGLE_KEY
    4) Env var: KAGGLE_API_TOKEN

This script detects any of those and, if absent, prints clear setup
instructions instead of bailing out silently.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


DATASETS = [
    ("rchitic17/real-or-fake", "data/raw/real_or_fake"),
    ("jruvika/fake-news-detection", "data/raw/fake_news_detection"),
]


# ----------------------------------------------------------------------
#  credential detection
# ----------------------------------------------------------------------

def _kaggle_dir() -> Path:
    """Resolve the directory where Kaggle looks for credentials."""
    env = os.environ.get("KAGGLE_CONFIG_DIR")
    if env:
        return Path(env)
    return Path.home() / ".kaggle"


def _have_kaggle_cli() -> bool:
    if shutil.which("kaggle") is not None:
        return True
    try:
        import kaggle  # noqa: F401
        return True
    except Exception:
        return False


def _have_kaggle_creds() -> tuple[bool, str]:
    """Return ``(ok, reason)`` describing which auth source worked."""
    kdir = _kaggle_dir()
    kaggle_json = kdir / "kaggle.json"
    access_token = kdir / "access_token"

    if kaggle_json.exists():
        return True, f"kaggle.json at {kaggle_json}"
    if access_token.exists() and access_token.stat().st_size > 0:
        return True, f"access_token at {access_token}"
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True, "KAGGLE_USERNAME + KAGGLE_KEY env vars"
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True, "KAGGLE_API_TOKEN env var"
    return False, ""


# ----------------------------------------------------------------------
#  download
# ----------------------------------------------------------------------

def _download_one(slug: str, dest: str) -> None:
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    print(f"[download] {slug} -> {dest_path}")
    # Prefer the CLI when available -- it streams progress nicely.
    if shutil.which("kaggle") is not None:
        subprocess.check_call(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest_path), "--force"]
        )
    else:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(slug, path=str(dest_path), force=True, quiet=False)
    for zf in dest_path.glob("*.zip"):
        print(f"[unzip] {zf.name}")
        with zipfile.ZipFile(zf) as z:
            z.extractall(dest_path)
        zf.unlink()


def _print_setup_help() -> None:
    print(
        "Kaggle credentials not found.\n"
        "\n"
        "On Windows PowerShell, set them up with EITHER of these:\n"
        "\n"
        "  -- Option A: classic kaggle.json --\n"
        "     1. Visit https://www.kaggle.com/settings -> 'Create New Token' (downloads kaggle.json)\n"
        "     2. Move the file:\n"
        "          New-Item -ItemType Directory -Force -Path $env:USERPROFILE\\.kaggle | Out-Null\n"
        "          Move-Item .\\kaggle.json $env:USERPROFILE\\.kaggle\\kaggle.json -Force\n"
        "\n"
        "  -- Option B: KGAT access_token --\n"
        "     New-Item -ItemType Directory -Force -Path $env:USERPROFILE\\.kaggle | Out-Null\n"
        "     Set-Content -Path $env:USERPROFILE\\.kaggle\\access_token "
        "-Value 'KGAT_xxxxxxxxxxxxxxxx' -Encoding ascii -NoNewline\n"
        "\n"
        "  -- Option C: env vars (one-off, this shell only) --\n"
        "     $env:KAGGLE_USERNAME = 'your-username'\n"
        "     $env:KAGGLE_KEY      = 'KGAT_xxxxxxxxxxxxxxxx'\n"
        "\n"
        "Then re-run:  python scripts/download_data.py\n"
        "\n"
        "If you cannot use the CLI, drop the CSVs manually at:\n"
        "  data/raw/real_or_fake/fake_or_real_news.csv\n"
        "  data/raw/fake_news_detection/data.csv\n"
    )


def main() -> int:
    have_cli = _have_kaggle_cli()
    have_creds, reason = _have_kaggle_creds()
    if not have_cli:
        print("Kaggle CLI not installed. Run:  pip install kaggle")
        return 1
    if not have_creds:
        _print_setup_help()
        return 1

    print(f"Using credentials: {reason}")
    for slug, dest in DATASETS:
        _download_one(slug, dest)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
