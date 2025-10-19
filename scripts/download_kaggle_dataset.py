"""Download a Kaggle dataset and extract it to the project `data/raw/` directory.

This script prefers the `kaggle` CLI (recommended) and falls back to the
Python `kaggle` package (KaggleApi). It expects Kaggle credentials to be set in
`~/.kaggle/kaggle.json` or via the environment variables `KAGGLE_USERNAME` and
`KAGGLE_KEY`.

Example usage:
  python scripts/download_kaggle_dataset.py \
      --dataset sharjeelmazhar/human-activity-recognition-video-dataset \
      --output_dir data/raw

If you don't have a Kaggle account configured, follow:
  https://github.com/Kaggle/kaggle-api#api-credentials
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def load_dotenv_file(env_path: Path):
    """Load simple KEY=VALUE pairs from a .env file into os.environ if not already set.

    This is a minimal parser (no quotes/escapes) sufficient for local development.
    """
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"')
            # don't overwrite existing environment variables
            if key and os.environ.get(key) is None:
                os.environ[key] = val
    except Exception:
        # Fail silently; credentials check will handle missing keys
        pass


def has_kaggle_cli() -> bool:
    return shutil.which("kaggle") is not None


def run_kaggle_cli(dataset: str, out_dir: Path, force: bool = False) -> int:
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(out_dir),
        "--unzip",
    ]
    if force:
        cmd.append("--force")
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def run_kaggle_api(dataset: str, out_dir: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError("Python kaggle package not available; please `pip install kaggle` or install kaggle CLI") from e

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading dataset {dataset} to {out_dir} (this may take a while)...")
    # KaggleApi downloads a zip file then extracts if unzip=True
    api.dataset_download_files(dataset, path=str(out_dir), unzip=True, quiet=False)


def assert_kaggle_credentials():
    # Check for ~/.kaggle/kaggle.json or env vars
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sharjeelmazhar/human-activity-recognition-video-dataset")
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument("--force", action="store_true", help="force re-download if files present")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load .env automatically if present
    load_dotenv_file(Path('.env'))

    if not assert_kaggle_credentials():
        print("Kaggle credentials not found. Create ~/.kaggle/kaggle.json or set KAGGLE_USERNAME and KAGGLE_KEY env vars.")
        print("See: https://github.com/Kaggle/kaggle-api#api-credentials")
        sys.exit(1)

    # If the directory already contains files and not forcing, skip
    if any(out_dir.iterdir()) and not args.force:
        print(f"Output directory {out_dir} is not empty. Use --force to re-download.")
        return

    # Prefer kaggle CLI
    if has_kaggle_cli():
        rc = run_kaggle_cli(args.dataset, out_dir, force=args.force)
        if rc != 0:
            print(f"kaggle CLI returned non-zero exit code {rc}. Trying Python Kaggle API if available.")
            try:
                run_kaggle_api(args.dataset, out_dir)
            except Exception as e:
                print("Download failed via both kaggle CLI and KaggleApi:", e)
                sys.exit(1)
    else:
        print("kaggle CLI not found; trying Python Kaggle API (pip install kaggle if missing).")
        try:
            run_kaggle_api(args.dataset, out_dir)
        except Exception as e:
            print("Download failed via KaggleApi:", e)
            print("Install kaggle CLI ('pip install kaggle' then 'kaggle --version') or configure credentials.")
            sys.exit(1)

    print("Download completed. Check the files under", out_dir)


if __name__ == "__main__":
    main()
