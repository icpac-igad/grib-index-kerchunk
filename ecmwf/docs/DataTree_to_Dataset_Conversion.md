# DataTree to Dataset Conversion Methods

## Understanding xarray DataTree Structure

### What is DataTree?
DataTree is a hierarchical data structure in xarray that organizes multiple datasets in a tree-like structure. Each node can contain:
- Dimensions
- Coordinates
- Data variables
- Attributes
- Child nodes (groups)

### Your Current DataTree Structure
```python
# From your example
dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)
# Result: 39 groups with only 2 time steps (0, 3)

# Structure:
DataTree
├── Groups: (39)
├── asn/instant/surface
│   ├── Dimensions: time=1, step=2, latitude=721, longitude=1440
│   └── Variables: asn, number, surface, valid_time
└── ... (other groups)
```

## Method 1: Convert Entire DataTree to Single Dataset

### Basic Conversion
```python
def datatree_to_dataset(dt):
    """Convert DataTree to single Dataset by flattening hierarchy."""
    import xarray as xr

    # Method 1a: Using to_dataset() if available
    try:
        ds = dt.to_dataset()
        return ds
    except AttributeError:
        pass

    # Method 1b: Manual flattening
    datasets = []

    def flatten_tree(node, path=""):
        """Recursively flatten the tree."""
        # Get dataset at current node
        if hasattr(node, 'ds') and node.ds is not None:
            ds = node.ds.copy()
            # Add path prefix to variable names
            if path:
                ds = ds.rename({var: f"{path}_{var}" for var in ds.data_vars})
            datasets.append(ds)

        # Process children
        if hasattr(node, 'children'):
            for name, child in node.children.items():
                child_path = f"{path}_{name}" if path else name
                flatten_tree(child, child_path)

    flatten_tree(dt)

    # Merge all datasets
    if datasets:
        return xr.merge(datasets)
    return xr.Dataset()
```

### Usage Example
```python
# Convert your DataTree to Dataset
ds = datatree_to_dataset(dt)
print(ds)

# Now you can work with it as a normal xarray Dataset
temperature = ds['asn_instant_surface_asn']  # Flattened variable name
```

## Method 2: Access Specific Groups/Nodes

### Direct Group Access
```python
def access_datatree_group(dt, path):
    """Access specific group in DataTree."""

    # Method 2a: Using path string
    groups = path.split('/')
    node = dt

    for group in groups:
        if hasattr(node, 'children') and group in node.children:
            node = node.children[group]
        elif hasattr(node, group):
            node = getattr(node, group)
        else:
            raise KeyError(f"Group '{group}' not found in path '{path}'")

    return node

# Example usage
surface_data = access_datatree_group(dt, 'asn/instant/surface')
ds_surface = surface_data.ds  # Get the dataset at this node
```

### Using Dictionary-Style Access
```python
# Method 2b: Dictionary-style access (if supported)
try:
    # Access nested groups
    surface_group = dt['asn']['instant']['surface']
    ds = surface_group.ds

    # Or using path
    surface_group = dt['asn/instant/surface']
    ds = surface_group.ds
except:
    # Fallback to attribute access
    surface_group = dt.asn.instant.surface
    ds = surface_group.ds
```

## Method 3: Selective Dataset Extraction

### Extract Specific Variables
```python
def extract_variables_from_datatree(dt, variable_names):
    """Extract specific variables from all groups in DataTree."""

    extracted_vars = {}

    def search_tree(node, path=""):
        """Recursively search for variables."""
        if hasattr(node, 'ds') and node.ds is not None:
            for var in variable_names:
                if var in node.ds.data_vars:
                    key = f"{path}_{var}" if path else var
                    extracted_vars[key] = node.ds[var]

        if hasattr(node, 'children'):
            for name, child in node.children.items():
                child_path = f"{path}_{name}" if path else name
                search_tree(child, child_path)

    search_tree(dt)

    # Create new dataset with extracted variables
    return xr.Dataset(extracted_vars)

# Usage
variables_needed = ['asn', 'tp', 't2m', 'u10', 'v10']
ds_extracted = extract_variables_from_datatree(dt, variables_needed)
```

## Method 4: Convert to Multi-Dataset Dictionary

### Create Dictionary of Datasets
```python
def datatree_to_dict(dt):
    """Convert DataTree to dictionary of datasets."""

    datasets = {}

    def traverse(node, path=""):
        """Traverse tree and collect datasets."""
        # Store dataset if exists
        if hasattr(node, 'ds') and node.ds is not None and len(node.ds.data_vars) > 0:
            datasets[path] = node.ds

        # Traverse children
        if hasattr(node, 'children'):
            for name, child in node.children.items():
                child_path = f"{path}/{name}" if path else name
                traverse(child, child_path)

    traverse(dt)
    return datasets

# Usage
ds_dict = datatree_to_dict(dt)
print(f"Found {len(ds_dict)} datasets:")
for path, ds in ds_dict.items():
    print(f"  {path}: {list(ds.data_vars)}")
```

## Method 5: Working with Incomplete Time Steps

### Expand Time Steps Using Index Files
```python
def expand_timesteps_in_datatree(dt, all_hours):
    """Expand DataTree from 2 steps to all 85 steps."""

    # Get existing structure from 0h and 3h
    template_ds = dt['asn/instant/surface'].ds

    # Create new dataset with all time steps
    new_steps = all_hours  # [0, 3, 6, ..., 360]

    # Expand step dimension
    expanded_ds = template_ds.reindex(step=new_steps, fill_value=np.nan)

    # Fill with data from index files
    for hour in all_hours:
        if hour not in [0, 3]:  # Skip existing
            # Load data for this hour from index
            hour_data = load_from_index(hour)
            expanded_ds.loc[dict(step=hour)] = hour_data

    return expanded_ds
```

## Method 6: Merge DataTree with Additional Data

### Combine Existing DataTree with New Time Steps
```python
def merge_datatree_timesteps(dt_existing, new_timesteps_data):
    """Merge existing DataTree (2 steps) with additional time steps."""

    # Convert existing to dataset
    ds_base = datatree_to_dataset(dt_existing)

    # For each new timestep
    for hour, hour_data in new_timesteps_data.items():
        # Create dataset for this hour
        ds_hour = create_dataset_from_references(hour_data)

        # Expand step dimension
        ds_hour = ds_hour.expand_dims(step=[hour])

        # Concatenate along step dimension
        if 'step' in ds_base.dims:
            ds_base = xr.concat([ds_base, ds_hour], dim='step')
        else:
            ds_base = ds_hour

    return ds_base
```

## Practical Examples for Your Use Case

### Example 1: Extract Surface Variables
```python
# Access surface group
surface = dt['asn']['instant']['surface']
ds_surface = surface.ds

# Extract specific variable
asn_data = ds_surface['asn']
print(f"ASN shape: {asn_data.shape}")
print(f"Time steps available: {ds_surface.step.values}")  # [0, 3]
```

### Example 2: Combine All Groups into Single Dataset
```python
def combine_all_groups(dt):
    """Combine all DataTree groups into single dataset."""

    all_data = {}

    # Walk through tree
    for path, node in dt.items():
        if hasattr(node, 'ds') and node.ds:
            # Prefix variables with path
            for var in node.ds.data_vars:
                key = f"{path.replace('/', '_')}_{var}"
                all_data[key] = node.ds[var]

    return xr.Dataset(all_data)

# Use it
full_ds = combine_all_groups(dt)
print(f"Combined dataset variables: {list(full_ds.data_vars)}")
```

### Example 3: Fix Incomplete Time Steps
```python
def fix_incomplete_timesteps(dt, target_hours):
    """Fix DataTree with only 2 time steps to have all 85."""

    # Get one group as template
    template = dt['asn']['instant']['surface'].ds

    # Current steps
    current_steps = template.step.values  # [0, 3]

    # Create expanded dataset
    expanded = xr.Dataset()

    for var in template.data_vars:
        # Get variable data
        var_data = template[var]

        # Create full array with NaN for missing steps
        full_shape = list(var_data.shape)
        step_idx = template.dims.index('step')
        full_shape[step_idx] = len(target_hours)

        full_data = np.full(full_shape, np.nan)

        # Fill in existing data
        for i, step in enumerate(current_steps):
            if step in target_hours:
                idx = target_hours.index(step)
                full_data[..., idx, ...] = var_data[..., i, ...]

        # Add to expanded dataset
        expanded[var] = (var_data.dims, full_data)

    # Set coordinates
    expanded = expanded.assign_coords(step=target_hours)

    return expanded
```

## Best Practices

### 1. Check DataTree Structure First
```python
def inspect_datatree(dt):
    """Inspect DataTree structure."""
    print(f"Total groups: {len(dt.groups())}")
    print("\nGroup paths:")
    for path in dt.groups():
        print(f"  {path}")

    print("\nVariables by group:")
    for path, node in dt.items():
        if hasattr(node, 'ds') and node.ds:
            vars = list(node.ds.data_vars)
            if vars:
                print(f"  {path}: {vars}")
```

### 2. Handle Missing Data Gracefully
```python
def safe_extract(dt, path, variable, default=None):
    """Safely extract variable from DataTree."""
    try:
        node = dt[path]
        if hasattr(node, 'ds') and variable in node.ds:
            return node.ds[variable]
    except:
        pass
    return default
```

### 3. Memory-Efficient Processing
```python
def process_datatree_lazy(dt):
    """Process DataTree using lazy loading."""
    # Don't load all data at once
    datasets = []

    for path, node in dt.items():
        if hasattr(node, 'ds') and node.ds:
            # Keep as lazy arrays
            ds_lazy = node.ds.chunk({'latitude': 100, 'longitude': 100})
            datasets.append(ds_lazy)

    # Merge lazily
    return xr.merge(datasets, compat='override')
```

## Troubleshooting Common Issues

### Issue: AttributeError when accessing groups
```python
# Solution: Use try-except with multiple access methods
def robust_access(dt, path):
    # Try different access methods
    try:
        return dt[path]  # Dictionary style
    except:
        try:
            parts = path.split('/')
            node = dt
            for part in parts:
                node = getattr(node, part)  # Attribute style
            return node
        except:
            return None
```

### Issue: Memory overflow with large DataTrees
```python
# Solution: Process in chunks
def process_large_datatree(dt, chunk_size=10):
    groups = list(dt.groups())
    results = []

    for i in range(0, len(groups), chunk_size):
        chunk_groups = groups[i:i+chunk_size]
        chunk_data = process_chunk(dt, chunk_groups)
        results.append(chunk_data)

    return xr.concat(results, dim='group')
```

## Summary

DataTree provides hierarchical organization but can be converted to standard xarray Datasets using:
1. **Flattening** - Combine all groups into single dataset
2. **Selection** - Extract specific groups or variables
3. **Dictionary** - Convert to dict of datasets
4. **Expansion** - Add missing time steps
5. **Merging** - Combine with additional data

For your ECMWF case with incomplete time steps (only 0h and 3h), the solution is to:
1. Use the index-based approach to get references for all 85 hours
2. Expand the existing DataTree structure
3. Fill in the missing time steps
4. Save as complete parquet files