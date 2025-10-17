# NOAA Climate Forecast System (CFS) Data Structure Analysis

## Overview
Comprehensive analysis of CFS data organization in AWS S3 bucket `s3://noaa-cfs-pds/` focusing on flux files (flxf) containing our target climate variables.

## Directory Structure

### Base Pattern
```
s3://noaa-cfs-pds/cfs.{YYYYMMDD}/{RUN}/
```

### Available Runs
- **00Z**: 4 daily runs with complete data
- **06Z**: 4 daily runs with complete data  
- **12Z**: 4 daily runs with complete data
- **18Z**: 4 daily runs with complete data

### Subdirectories per Run
```
{RUN}/
├── 6hrly_grib_01/     # Primary 6-hourly data (longest forecast)
├── 6hrly_grib_02/     # Secondary ensemble member
├── 6hrly_grib_03/     # Secondary ensemble member  
├── 6hrly_grib_04/     # Secondary ensemble member
├── monthly_grib_01/   # Monthly aggregated data
├── monthly_grib_02/   # (00Z only - others vary)
├── monthly_grib_03/   # (00Z only)
├── monthly_grib_04/   # (00Z only)
├── time_grib_01/      # Time series data
├── time_grib_02/      # Time series data
├── time_grib_03/      # Time series data
└── time_grib_04/      # Time series data
```

## Flux File (flxf) Analysis

### File Naming Pattern
```
flxf{YYYYMMDDHH}.{MEMBER}.{INIT_DATE}.grb2
flxf{YYYYMMDDHH}.{MEMBER}.{INIT_DATE}.grb2.idx
```

Where:
- `{YYYYMMDDHH}`: Forecast valid time (year, month, day, hour)
- `{MEMBER}`: Ensemble member (01, 02, 03, 04)
- `{INIT_DATE}`: Model initialization date

### Flux File Counts by Directory

#### 00Z Run (2025-07-01)
| Directory | flxf Files | Forecast Period | Duration |
|-----------|------------|-----------------|----------|
| 6hrly_grib_01 | 1,722 | 2025-07-01 00Z → 2026-02-01 00Z | **215 days** (7.1 months) |
| 6hrly_grib_02 | 986 | 2025-07-01 00Z → 2025-11-01 00Z | **123 days** (4.1 months) |
| 6hrly_grib_03 | 986 | 2025-07-01 00Z → 2025-11-01 00Z | **123 days** (4.1 months) |
| 6hrly_grib_04 | 986 | 2025-07-01 00Z → 2025-11-01 00Z | **123 days** (4.1 months) |

#### Other Runs (6hrly_grib_01 only)
| Run | flxf Files | Expected Duration |
|-----|------------|------------------|
| 06Z | 1,720 | ~215 days |
| 12Z | 1,718 | ~215 days |
| 18Z | 1,716 | ~215 days |

### File Pairing Verification
✅ **Each GRIB2 file has corresponding .idx file confirmed**

Example from 6hrly_grib_01:
```
flxf2025070100.01.2025070100.grb2      (4.3 MB)
flxf2025070100.01.2025070100.grb2.idx  (4.9 KB)
```

## Forecast Length Analysis

### Primary Member (grib_01)
- **All runs extend ~7 months (215+ days)**
- Start: Model initialization date
- End: Approximately 7 months from start
- Temporal resolution: **6-hourly intervals**

### Secondary Members (grib_02-04) 
- **Limited to ~4 months (123 days)**
- Same start date as primary
- Shorter forecast horizon
- Same temporal resolution: **6-hourly intervals**

### Calculation Details
For 6-hourly data over 215 days:
- Total timesteps: 215 days × 4 timesteps/day = 860 timesteps
- Plus initial conditions ≈ **861 files** per member
- Observed: 1,722 files (861 × 2 for .grb2 + .idx)

## Variable Availability in Flux Files

### Confirmed Variables in flxf Files
From analysis of sample index file:
1. **PRATE** - Precipitation rate (surface)
2. **TMP** - Temperature at 2m above ground  
3. **DSWRF** - Downward shortwave radiation (surface)
4. **USWRF** - Upward shortwave radiation (surface)
5. **DLWRF** - Downward longwave radiation (surface)
6. **ULWRF** - Upward longwave radiation (surface) 
7. **UGRD** - U-component wind at 10m above ground
8. **VGRD** - V-component wind at 10m above ground

### Missing Variables
- **TMIN/TMAX**: Not directly available, must be computed from 6-hourly TMP data

## Implementation Implications

### For Kerchunk Processing

1. **Primary Focus**: Use `6hrly_grib_01` for longest forecast range
2. **Ensemble Processing**: Can include `6hrly_grib_02-04` for shorter-range ensemble analysis
3. **Temporal Range**: 
   - Short-term: 0-123 days (all 4 members)
   - Long-term: 0-215 days (member 01 only)

### URL Generation Pattern
```python
def cfs_flux_url_generator(date_str, run, member, forecast_datetime):
    """
    Generate CFS flux file URLs
    
    Args:
        date_str: Init date (YYYYMMDD)
        run: Model run (00, 06, 12, 18)
        member: Ensemble member (01, 02, 03, 04)
        forecast_datetime: Target forecast time
    """
    forecast_str = forecast_datetime.strftime("%Y%m%d%H")
    return f"s3://noaa-cfs-pds/cfs.{date_str}/{run}/6hrly_grib_{member}/flxf{forecast_str}.{member}.{date_str}{run}.grb2"
```

### Processing Strategy
1. **Start with member 01** (longest range, 215 days)
2. **Validate variable extraction** from flux files
3. **Scale to multiple members** for ensemble analysis
4. **Implement TMIN/TMAX computation** from 6-hourly temperature

## Data Volume Estimates

### Per Member (grib_01)
- GRIB2 files: ~861 files × 4MB = **~3.4 GB**
- Index files: ~861 files × 5KB = **~4.3 MB**
- Total per member: **~3.4 GB**

### Full 4-Member Ensemble (123 days)
- Total data volume: **~13.6 GB**
- Processing complexity: Moderate (multiple S3 buckets)

### Long-range Single Member (215 days)
- Member 01 only: **~3.4 GB**
- Recommended for initial implementation

## CFS Ensemble Strategy: Emulating ECMWF CDS Structure

### 4.1 CFS Forecast Structure (ECMWF CDS Reference)
The CFS dataset available through ECMWF Climate Data Store has this structure:

**Dimensions:**
- `number`: 124 (31 days × 4 cycles per day)
- `forecastMonth`: 6 (forecast months) 
- `time`: 604 (forecast time steps)
- `latitude`: 36 (grid cells in latitude)
- `longitude`: 33 (grid cells in longitude)

**Variables:**
- `tprate` (precipitation rate) + other meteorological variables

### Ensemble Construction Strategy

#### Option 1: Multi-Date Ensemble (ECMWF CDS Style)
**Concept**: Create ensemble by combining forecasts from different initialization dates
```
For target month (e.g., August 2025):
- Use 31 dates × 4 runs = 124 ensemble members
- Date range: July 1-31, 2025 (all runs: 00Z, 06Z, 12Z, 18Z)
- Each member forecasts the same target period (August 2025)
- Combines forecast uncertainty with initial condition uncertainty
```

**Implementation Pattern:**
```python
ensemble_members = []
for date in july_dates:  # 31 dates
    for run in ["00", "06", "12", "18"]:  # 4 runs per day
        member_id = f"{date}_{run}"
        forecast_files = get_cfs_flux_files(date, run, target_month="august")
        ensemble_members.append((member_id, forecast_files))

# Result: 124 ensemble members for August 2025 forecasts
```

#### Option 2: Multi-Member Ensemble (AWS Native)
**Concept**: Use AWS CFS native ensemble structure
```
For target period:
- Use 4 ensemble members (grib_01, grib_02, grib_03, grib_04)
- Same initialization date
- Different model perturbations
- Shorter ensemble (4 members vs 124)
```

### Recommended Hybrid Approach

#### Phase 1: Multi-Date Ensemble (Preferred)
1. **Target forecast period**: 6 months ahead
2. **Ensemble construction**: 
   - **124 members**: 31 recent dates × 4 daily runs
   - **Forecast range**: Focus on 6-month horizon matching ECMWF CDS
   - **Data source**: Primarily `6hrly_grib_01` (longest range)

3. **Monthly ensemble example**:
```
Target: September 2025 climate forecast
Ensemble members:
- 2025-08-01 00Z → Sep 2025 forecast (member 001)
- 2025-08-01 06Z → Sep 2025 forecast (member 002)
- 2025-08-01 12Z → Sep 2025 forecast (member 003)
- 2025-08-01 18Z → Sep 2025 forecast (member 004)
- 2025-08-02 00Z → Sep 2025 forecast (member 005)
...
- 2025-08-31 18Z → Sep 2025 forecast (member 124)
```

#### Phase 2: Extended Multi-Member
1. **Include grib_02-04** for recent dates (123-day range)
2. **Potential ensemble size**: 124×4 = 496 members for near-term forecasts
3. **Use case**: Sub-seasonal to seasonal prediction

### Updated Implementation Strategy

#### Ensemble Dimension Mapping
```python
# ECMWF CDS style dimensions
cfs_ensemble_dims = {
    'number': 124,        # 31 days × 4 cycles
    'forecastMonth': 6,   # 6-month forecast horizon  
    'time': 'variable',   # Depends on temporal resolution
    'latitude': 'native', # CFS native grid
    'longitude': 'native' # CFS native grid
}
```

#### URL Generation for Multi-Date Ensemble
```python
def generate_cfs_ensemble_urls(target_month, init_period_days=31):
    """
    Generate CFS ensemble URLs following ECMWF CDS pattern
    
    Args:
        target_month: Target forecast month
        init_period_days: Days of initialization dates (default: 31)
    
    Returns:
        List of URLs for 124 ensemble members
    """
    ensemble_urls = []
    
    # Calculate initialization period (e.g., July for August forecast)
    init_start = target_month - timedelta(days=init_period_days)
    
    for day in range(init_period_days):
        init_date = init_start + timedelta(days=day)
        for run in ["00", "06", "12", "18"]:
            member_urls = get_cfs_flux_files_for_period(
                init_date.strftime("%Y%m%d"), 
                run, 
                forecast_period_months=6
            )
            ensemble_urls.append({
                'member_id': f"{init_date.strftime('%Y%m%d')}_{run}",
                'init_date': init_date,
                'run': run,
                'urls': member_urls
            })
    
    return ensemble_urls  # 124 members
```

### Data Volume for Multi-Date Ensemble
- **124 members × 6 months each**: ~420 GB total
- **Per target month**: ~70 GB
- **Recommended**: Process monthly chunks, store as kerchunk references

### Advantages of Multi-Date Ensemble
1. **Climate prediction focus**: Better captures seasonal variability
2. **ECMWF CDS compatibility**: Same ensemble structure and size
3. **Uncertainty representation**: Combines forecast and initial condition uncertainty
4. **Operational relevance**: Mimics real-world seasonal forecasting practices

## Next Steps for Implementation

1. **Implement multi-date ensemble strategy** following ECMWF CDS pattern
2. **Focus on 6-month forecast horizon** for climate applications
3. **Build 124-member ensemble** from 31 days × 4 runs
4. **Use `6hrly_grib_01`** as primary data source
5. **Validate against ECMWF CDS** structure and dimensions
6. **Consider monthly processing** to manage data volume
7. **Scale processing** for multiple target months/seasons