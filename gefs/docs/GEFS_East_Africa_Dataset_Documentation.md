# GEFS East Africa 30 Ensemble Members Dataset Documentation

## Overview

This document describes the 30 ensemble member dataset created for the East Africa region in the GEFS (Global Ensemble Forecast System) precipitation analysis pipeline. The dataset is specifically configured for regional weather forecasting and empirical probability calculations in the `run_gefs_24h_accumulation.py` script.

## Geographic Coverage

The East Africa region is defined with the following spatial boundaries:

- **Latitude Range**: -12° to 23° N (covering from southern Tanzania to northern Sudan/Ethiopia)
- **Longitude Range**: 21° to 53° E (covering from eastern Chad to Somalia/eastern Ethiopia)
- **Reference Time Zone**: East Africa Time (UTC+3)

### Key Geographic Features Covered
- **Countries**: Kenya, Tanzania, Uganda, Rwanda, Burundi, South Sudan, Ethiopia, Somalia, eastern Sudan, eastern Chad, eastern DRC
- **Major Cities**: Nairobi (marked as reference point at -1.2921°N, 36.8219°E)
- **Climate Zones**: Tropical highland, arid/semi-arid, equatorial rainforest margins

## Dataset Structure

### Ensemble Configuration
- **Total Members**: 30 ensemble members (gep01 through gep30)
- **Control Run**: gep00 (operational control forecast)
- **Perturbed Members**: gep01-gep30 (probabilistic ensemble members)

### Temporal Resolution
- **Forecast Timestep**: 3-hourly intervals
- **Forecast Length**: 240 hours (10 days)
- **24-hour Periods**: 10 complete days from forecast initialization
- **Timesteps per Day**: 8 (for 24-hour accumulation calculations)

### Data Processing Workflow

#### 1. Data Ingestion
```python
# Each ensemble member stored as Kerchunk reference in parquet format
parquet_files = sorted(PARQUET_DIR.glob("gep*.par"))
# Regional extraction during streaming
regional_data = data_var.sel(
    latitude=slice(LAT_MAX, LAT_MIN),    # 23° to -12°
    longitude=slice(LON_MIN, LON_MAX)    # 21° to 53°
)
```

#### 2. 24-Hour Accumulation Processing
```python
# Skip timestep 0 (initial condition), use forecast timesteps 1-80
forecast_data = data[1:]  # timesteps 1-80 contain 3-hourly precipitation
# Sum 8 consecutive timesteps for 24-hour totals
for day in range(n_days):
    start_idx = day * TIMESTEPS_PER_DAY
    end_idx = (day + 1) * TIMESTEPS_PER_DAY
    daily_accum[day] = np.sum(forecast_data[start_idx:end_idx], axis=0)
```

#### 3. Empirical Probability Calculation
The system calculates exceedance probabilities using all 30 ensemble members:

```python
# Stack all member data for each day
day_stack = np.stack(day_data, axis=0)  # Shape: (30, lat, lon)
# Calculate probability of exceeding threshold
exceedance_count = np.sum(day_stack >= threshold, axis=0)
probability = (exceedance_count / 30) * 100  # Convert to percentage
```

### Precipitation Thresholds
The system analyzes exceedance probabilities for the following 24-hour rainfall thresholds:
- **5mm**: Light precipitation threshold
- **25mm**: Moderate precipitation threshold  
- **50mm**: Heavy precipitation threshold
- **75mm**: Very heavy precipitation threshold
- **100mm**: Extreme precipitation threshold
- **125mm**: Exceptional precipitation threshold

## Technical Implementation Details

### Data Storage Format
- **Input Format**: Kerchunk parquet files with S3 references
- **Processing Format**: NumPy arrays for computation, xarray for coordinate handling
- **Variable**: Total precipitation (`tp`) in mm (kg/m²)

### Memory Management
- **Streaming Approach**: Each ensemble member processed individually to manage memory
- **Regional Subsetting**: Data extracted for East Africa region only during loading
- **Computational Efficiency**: 24-hour accumulations calculated in vectorized operations

### Quality Control
- **Missing Data Handling**: Initial timestep (t=0) skipped as it contains NaN values
- **Temporal Consistency**: Ensures 8 complete timesteps available for each 24-hour period
- **Spatial Validation**: Regional bounds validated against coordinate arrays

## Output Products

### Probability Maps
The system generates comprehensive probability visualizations:

1. **Multi-threshold Probability Plots**: Shows exceedance probabilities for all thresholds across multiple forecast days
2. **Individual Threshold Maps**: Detailed maps for specific precipitation thresholds
3. **Summary Statistics**: Maximum probabilities and spatial coverage metrics

### Forecast Timing
All forecasts reference East Africa Time (UTC+3) for operational relevance:
- **Model Run Time**: Converted from UTC to EAT in all displays
- **Forecast Valid Time**: 24-hour periods aligned to EAT daily cycles
- **Probability Windows**: Each 24-hour accumulation period clearly labeled with EAT timestamps

## Applications

### Operational Weather Forecasting
- **Daily Precipitation Outlook**: 10-day probabilistic rainfall forecast
- **Extreme Event Early Warning**: High-probability areas for heavy precipitation
- **Agricultural Planning**: Seasonal and multi-day precipitation guidance

### Climate Risk Assessment
- **Drought Monitoring**: Low probability areas for minimum thresholds
- **Flood Risk**: High probability areas for extreme thresholds
- **Water Resource Management**: Regional precipitation distribution analysis

### Research Applications
- **Model Validation**: Ensemble spread and reliability analysis
- **Climate Studies**: Long-term precipitation pattern analysis
- **Hydrological Modeling**: Input for watershed and river basin studies

## Performance Characteristics

Based on processing benchmarks in the script:
- **Data Loading**: ~30-60 seconds for 30 ensemble members
- **24h Accumulation**: ~10-20 seconds for regional processing
- **Probability Calculation**: ~5-10 seconds for all thresholds
- **Visualization Generation**: ~20-40 seconds for complete plot suite

## Future Enhancements

### Potential Improvements
1. **Higher Resolution**: Integration of higher spatial resolution ensemble data
2. **Extended Range**: Incorporation of sub-seasonal ensemble forecasts
3. **Additional Variables**: Temperature, wind, and other meteorological parameters
4. **Real-time Processing**: Automated ingestion and processing pipeline
5. **Interactive Visualization**: Web-based probability map interface

---

*This documentation corresponds to the East Africa ensemble dataset implementation in `run_gefs_24h_accumulation.py` as of the current version. The dataset provides a robust foundation for probabilistic precipitation forecasting across the East Africa region using 30 GEFS ensemble members.*