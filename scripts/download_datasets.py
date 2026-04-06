#!/usr/bin/env python3
"""Download all required datasets to .dataset/"""
import subprocess
import sys
from pathlib import Path

DATASET_DIR = Path(".dataset")
DATASET_DIR.mkdir(exist_ok=True)


def download_dabstep():
    """Download adyen/DABstep dataset."""
    csv_path = DATASET_DIR / "dabstep_data.csv"
    data_dir = DATASET_DIR / "DABstep-data"

    if csv_path.exists() and data_dir.exists():
        print(f"[skip] DABstep already exists at {csv_path}")
        return

    print("Downloading adyen/DABstep ...")
    from datasets import load_dataset
    import pandas as pd

    # Load the tasks config (contains questions, answers, guidelines)
    ds = load_dataset("adyen/DABstep", "tasks")
    split = "default"  # 450 tasks
    df = ds[split].to_pandas()
    df.to_csv(csv_path, index=False)
    print(f"  -> saved {len(df)} rows to {csv_path}")

    # Download the context/data files via git-lfs clone
    if not data_dir.exists():
        print("  Cloning DABstep-data repo for context files ...")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://huggingface.co/datasets/adyen/DABstep",
             str(data_dir)],
            check=True,
        )
        print(f"  -> context files at {data_dir}/data/context")
    else:
        print(f"[skip] DABstep data dir already exists at {data_dir}")


def download_sealqa():
    """Download vtllms/sealqa dataset."""
    csv_path = DATASET_DIR / "seal-0.csv"

    if csv_path.exists():
        print(f"[skip] sealqa already exists at {csv_path}")
        return

    print("Downloading vtllms/sealqa ...")
    from datasets import load_dataset
    import pandas as pd

    ds = load_dataset("vtllms/sealqa", "seal_0")
    df = ds["test"].to_pandas()
    df.to_csv(csv_path, index=False)
    print(f"  -> saved {len(df)} rows to {csv_path}")


def download_officeqa():
    """Download databricks/officeqa from GitHub."""
    officeqa_dir = DATASET_DIR / "officeqa"
    csv_path = officeqa_dir / "officeqa_pro.csv"

    if csv_path.exists():
        print(f"[skip] officeqa already exists at {csv_path}")
        return

    print("Cloning databricks/officeqa from GitHub ...")
    officeqa_dir.mkdir(exist_ok=True)
    tmp_repo = DATASET_DIR / "_officeqa_repo"

    if not tmp_repo.exists():
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/databricks/officeqa",
             str(tmp_repo)],
            check=True,
        )

    # Find and copy CSV files
    import shutil
    found = list(tmp_repo.rglob("officeqa_pro.csv"))
    if found:
        shutil.copy(found[0], csv_path)
        print(f"  -> copied officeqa_pro.csv to {csv_path}")
    else:
        # Show what's available
        all_csvs = list(tmp_repo.rglob("*.csv"))
        print(f"  [WARNING] officeqa_pro.csv not found. Available CSVs:")
        for f in all_csvs:
            print(f"    {f}")
        # Copy everything to officeqa_dir
        for f in all_csvs:
            dest = officeqa_dir / f.name
            shutil.copy(f, dest)
            print(f"  -> copied {f.name} to {dest}")

    # Copy any other data files (e.g. documents)
    for subdir in tmp_repo.iterdir():
        if subdir.is_dir() and subdir.name not in [".git"]:
            dest = officeqa_dir / subdir.name
            if not dest.exists():
                shutil.copytree(subdir, dest)
                print(f"  -> copied dir {subdir.name}/")


if __name__ == "__main__":
    import os
    # Run from EvoSkill root
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)

    print(f"Working dir: {Path.cwd()}")
    print(f"Target: {DATASET_DIR.resolve()}\n")

    download_dabstep()
    print()
    download_sealqa()
    print()
    download_officeqa()
    print("\nDone.")
