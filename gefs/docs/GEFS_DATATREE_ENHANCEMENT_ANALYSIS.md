# GEFS DataTree Enhancement Analysis

## Question: Can the current gribberish routine be enhanced with DataTree?

**Short Answer**: Yes, but with important tradeoffs to understand.

---

## 1. Understanding "Lazy Loading"

There are **two different types** of lazy loading to consider:

### Type A: Zarr Lazy Loading (Current Output)

**What we have now**: Individual zarr files per member (`gep01.zarr`, `gep02.zarr`, etc.)

```python
# Opening a zarr file is LAZY - only metadata loaded
ds = xr.open_dataset('gep01.zarr', engine='zarr')
# Memory: ~68 MB (metadata only)

# Accessing .values TRIGGERS actual data load
data = ds.t2m.values
# Memory: +5.6 MB (data now in RAM)

# With dask chunks - FULLY LAZY
ds = xr.open_dataset('gep01.zarr', chunks={'step': 10})
# Memory: minimal until .compute()
```

**Test Results** (from profiling):
```
After open_dataset(): 68 MB (metadata only)
After .values:        83 MB (data loaded)
With dask chunks:     Data NOT loaded until .compute()
```

### Type B: Reference-Based Lazy Loading

**Alternative approach**: Keep only byte references, decode on-demand

```python
# fsspec reference filesystem - data stays in S3
fs = fsspec.filesystem("reference", fo=zstore, remote_protocol='s3')
dt = xr.open_datatree(fs.get_mapper(""), engine="zarr")

# Data decoded ONLY when accessed
values = dt['gep01']['tp'].compute()  # Triggers S3 fetch + gribberish decode
```

---

## 2. Current Processing Pipeline Analysis

### What Happens Now (`run_single_gefs_to_zarr_gribberish.py`):

```
┌──────────────────────────────────────────────────────────────────┐
│                    CREATION TIME (NOT LAZY)                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Parquet refs ─▶ S3 fetch ─▶ Gribberish decode ─▶ Save to zarr   │
│  (offset/len)     (GRIB)      (to numpy array)     (decoded)      │
│                                                                   │
│  Time: ~28s per member (8 variables, 81 timesteps)               │
│  Memory: ~860 MB peak                                            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    ACCESS TIME (LAZY)                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  xr.open_dataset() ─▶ Access .values ─▶ Data in memory           │
│  (metadata only)       (triggers load)                           │
│                                                                   │
│  Time: milliseconds (data already decoded)                       │
│  Memory: Only what you access                                    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Key Point**: Gribberish decoding happens at **creation time**, not access time.

---

## 3. DataTree Enhancement Options

### Option A: Combine Individual Zarrs into DataTree (Post-Processing)

```python
# After batch processing completes
from xarray import DataTree

# Open all member zarrs
members = {}
for i in range(1, 31):
    member = f'gep{i:02d}'
    zarr_path = f'zarr_stores/20250918_00/{member}_gribberish_east_africa.zarr'
    members[member] = xr.open_dataset(zarr_path, engine='zarr')

# Create DataTree structure
dt = DataTree.from_dict({
    '/': xr.Dataset(attrs={'title': 'GEFS Ensemble'}),
    **{f'/{m}': ds for m, ds in members.items()},
})

# Add ensemble statistics (computed lazily with dask)
all_data = xr.concat([dt[f'/{m}'].ds for m in members], dim='member')
dt['/ensemble_stats'] = xr.Dataset({
    'mean': all_data.mean(dim='member'),
    'std': all_data.std(dim='member'),
})

# Save as single DataTree zarr
dt.to_zarr('ensemble_datatree.zarr')
```

**Pros**:
- Single file for all members
- Hierarchical access: `dt['gep01']['t2m']`
- Can include pre-computed statistics
- Still lazy on access

**Cons**:
- Extra post-processing step
- Larger single file to manage
- No speed improvement at creation time

### Option B: Reference-Based DataTree (True Lazy Decoding)

```python
# Keep GRIB references, decode only on access
combined_refs = {}
for member_parquet in parquet_files:
    zstore = read_parquet(member_parquet)
    for key, value in zstore.items():
        combined_refs[f'{member}/{key}'] = value

# Save combined references
save_combined_parquet(combined_refs, 'ensemble_refs.par')

# Access is lazy - gribberish decodes on demand
fs = fsspec.filesystem("reference", fo=combined_refs, ...)
dt = xr.open_datatree(fs.get_mapper(""), engine="zarr")
```

**Pros**:
- No upfront decoding (fast "creation")
- Minimal storage (just references)
- True lazy loading

**Cons**:
- Slow repeated access (decode every time)
- Requires network for each access
- Gribberish ~1% failure rate on access

### Option C: Hybrid - Pre-decode Common Data, Lazy for Rest

```python
# Pre-decode frequently accessed data (e.g., ensemble mean, first few timesteps)
# Keep references for rarely accessed data (individual members, late timesteps)
```

---

## 4. Recommendation

### For Most Use Cases: Keep Current Approach + Post-Processing DataTree

```
Batch Processing (current)     Post-Processing (new)
─────────────────────────     ─────────────────────
gep01.zarr  ┐                       │
gep02.zarr  │                       ▼
gep03.zarr  ├──────────────▶  ensemble_datatree.zarr
...         │                   /gep01
gep30.zarr  ┘                   /gep02
                                /...
                                /ensemble_stats
                                /probabilities
```

**Why**:
1. **Fast repeated access**: Data pre-decoded, millisecond access
2. **Organized structure**: Hierarchical access to members
3. **Pre-computed stats**: Ensemble mean, std, probabilities
4. **Memory efficient**: Still lazy-loads from zarr

### Implementation

Add to `run_batch_gefs_gribberish.py`:

```python
def create_ensemble_datatree(zarr_dir, output_path, members):
    """Combine individual zarrs into a single DataTree."""
    from xarray import DataTree

    # Open all members (lazy)
    member_datasets = {}
    for member in members:
        zarr_path = zarr_dir / f'{member}_gribberish_east_africa.zarr'
        if zarr_path.exists():
            member_datasets[member] = xr.open_dataset(zarr_path, engine='zarr')

    # Create tree structure
    tree_dict = {'/': xr.Dataset(attrs={
        'title': 'GEFS Ensemble Forecast',
        'n_members': len(member_datasets),
    })}

    for member, ds in member_datasets.items():
        tree_dict[f'/{member}'] = ds

    # Compute ensemble statistics (with dask for efficiency)
    stacked = xr.concat(list(member_datasets.values()), dim='member')
    tree_dict['/ensemble_stats'] = xr.Dataset({
        'mean': stacked.mean(dim='member'),
        'std': stacked.std(dim='member'),
        'min': stacked.min(dim='member'),
        'max': stacked.max(dim='member'),
    })

    dt = DataTree.from_dict(tree_dict)
    dt.to_zarr(output_path)

    return dt
```

---

## 5. Summary: When is Compute Happening?

| Stage | Compute? | Description |
|-------|----------|-------------|
| **Parquet creation** | No | Just byte references |
| **Gribberish batch processing** | **YES** | Decodes all GRIB data |
| **Zarr save** | No | Writes decoded arrays |
| **xr.open_dataset()** | No | Loads metadata only |
| **Access .values** | No* | Reads from disk |
| **With dask .compute()** | Delayed | Loads on demand |

*Data already decoded, just reading from zarr

---

## 6. Performance Comparison

| Approach | Creation Time | Access Time | Storage |
|----------|--------------|-------------|---------|
| **Current (individual zarr)** | 28s/member | <1s | 3.5MB/member |
| **Reference-only (no decode)** | <1s/member | 28s/access | ~0.1MB/member |
| **DataTree (post-process)** | +30s total | <1s | Same total |

**Bottom line**: If you access data multiple times, pre-decoding wins. If you access once, references might be faster.

---

## 7. Code Example: Testing Lazy Loading

```python
import xarray as xr
import psutil

# Open zarr (LAZY - only metadata)
ds = xr.open_dataset('gep01.zarr', engine='zarr')
print(f"After open: {psutil.Process().memory_info().rss / 1e6:.0f} MB")

# Access shape (still lazy)
print(f"Shape: {ds.t2m.shape}")
print(f"After shape: {psutil.Process().memory_info().rss / 1e6:.0f} MB")

# Load data (TRIGGERS READ)
data = ds.t2m.values
print(f"After .values: {psutil.Process().memory_info().rss / 1e6:.0f} MB")

# With dask (FULLY LAZY until compute)
ds_dask = xr.open_dataset('gep01.zarr', engine='zarr',
                          chunks={'step': 10})
print(f"Dask type: {type(ds_dask.t2m.data)}")  # dask.array
subset = ds_dask.t2m.isel(step=0).compute()  # Only loads 1 timestep
```

---

*Document created: 2025-12-08*
