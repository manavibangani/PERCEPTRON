"""Load the complete Criteo uplift dataset into a workspace-local cache."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "huggingface"))

from datasets import load_dataset

ds = load_dataset("criteo/criteo-uplift")

if __name__ == "__main__":
    print(ds)
    print("Columns:", ds["train"].column_names)
    print("First row:", ds["train"][0])
    print("Cache:", os.environ["HF_HOME"])
