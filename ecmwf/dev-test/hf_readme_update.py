#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub",
# ]
# ///
"""Update the HF README to add catalog section."""

from huggingface_hub import HfApi, hf_hub_download
import tempfile, os

api = HfApi()

# Download current README
path = hf_hub_download(repo_id="E4DRR/gik-ecmwf-par", repo_type="dataset",
                       filename="README.md")
readme = open(path).read()

# Insert catalog section after "## Dataset Structure" section
catalog_section = """## Catalog / Index

A lightweight **catalog.parquet** (1.7 MB) at the repo root indexes all 144,228 parquet files
across the dataset. Use it to discover available dates, runs, and members without listing the
full repo tree.

**Download**: [`catalog.parquet`](catalog.parquet)

| Column | Example | Description |
|--------|---------|-------------|
| `year` | `2024` | Forecast year |
| `month` | `03` | Forecast month |
| `date` | `20240301` | Forecast date (YYYYMMDD) |
| `run` | `00z` | Run hour |
| `member` | `control` | Ensemble member name |
| `filename` | `2024030100z-control.parquet` | Parquet filename |
| `hf_path` | `run_par_ecmwf/2024/03/20240301/00z/...` | Full path in this repo |
| `size_bytes` | `107520` | File size in bytes |

### Quick Start with the Catalog

```python
import pandas as pd
from huggingface_hub import hf_hub_download

# 1. Load the catalog (1.7 MB, indexes all 144k+ files)
catalog_path = hf_hub_download(
    repo_id="E4DRR/gik-ecmwf-par", repo_type="dataset",
    filename="catalog.parquet"
)
catalog = pd.read_parquet(catalog_path)

# 2. Explore what's available
print(catalog.groupby(["year", "month"]).size())       # files per month
print(catalog["run"].unique())                          # ['00z','06z','12z','18z']
print(catalog["member"].nunique())                      # 51

# 3. Filter for a specific date + run
subset = catalog[(catalog["date"] == "20250101") & (catalog["run"] == "00z")]
print(subset[["member", "filename", "size_bytes"]])

# 4. Download a specific parquet using its hf_path
row = subset.iloc[0]
parquet_path = hf_hub_download(
    repo_id="E4DRR/gik-ecmwf-par", repo_type="dataset",
    filename=row["hf_path"]
)
```

### Coverage Summary (from catalog)

| Year | Months | Dates | Files | Total Size |
|------|--------|-------|-------|------------|
| 2024 | Mar--Dec | 306 | ~62,424 | ~6.5 GB |
| 2025 | Jan--Dec | 365 | ~75,504 | ~7.8 GB |
| 2026 | Jan--Feb | 51 | ~7,344 | ~0.8 GB |
| **Total** | | **720** | **144,228** | **~17.8 GB** |

"""

# Find insertion point: after "## Dataset Structure" section, before "## How It Works"
marker = "## How It Works"
if marker in readme:
    readme = readme.replace(marker, catalog_section + marker)
    print("Inserted catalog section before '## How It Works'")
else:
    # Fallback: append before "## Project"
    readme = readme.replace("## Project", catalog_section + "## Project")
    print("Inserted catalog section before '## Project'")

# Upload updated README
tmpdir = tempfile.mkdtemp()
readme_path = os.path.join(tmpdir, "README.md")
with open(readme_path, "w") as f:
    f.write(readme)

api.upload_file(
    path_or_fileobj=readme_path,
    path_in_repo="README.md",
    repo_id="E4DRR/gik-ecmwf-par",
    repo_type="dataset",
)
print("README.md updated on HuggingFace")
