# GEFS Gribberish Processing: Profiling & Batch Processing Guide

## Overview

This document describes how to profile the GEFS gribberish processing script and determine optimal batch processing parameters for your system.

---

## 1. Profiling Methodology

### 1.1 Why Profile?

Before running batch processing with multiple parallel workers, you need to understand:
- **Memory usage per process**: How much RAM does each worker consume?
- **Peak vs steady-state memory**: Memory spikes during processing
- **CPU utilization**: How CPU-bound vs I/O-bound is the workload?
- **System capacity**: How many parallel workers can run safely?

### 1.2 Profiling Tools

| Tool | Purpose | Command |
|------|---------|---------|
| `ps` | Process memory snapshot | `ps aux \| grep gribberish` |
| `/proc/<pid>/status` | Detailed memory breakdown | `cat /proc/<pid>/status` |
| `psutil` (Python) | Programmatic monitoring | Built into batch script |
| `/proc/meminfo` | System memory status | `cat /proc/meminfo` |

### 1.3 Key Metrics to Measure

```bash
# 1. Check system resources
free -h
nproc

# 2. Monitor running process
PID=$(ps aux | grep gribberish | grep -v grep | awk '{print $2}')

# Get memory usage
ps -p $PID -o pid,vsz,rss,pmem,pcpu,etime

# Detailed memory from /proc
cat /proc/$PID/status | grep -E "VmSize|VmRSS|VmPeak|VmData"
```

---

## 2. Profiling Results (December 2025)

### 2.1 Test Configuration

| Parameter | Value |
|-----------|-------|
| **Script** | `run_single_gefs_to_zarr_gribberish.py` |
| **Data** | GEFS 20250918 00Z |
| **Region** | East Africa |
| **Variables** | t2m, tp, u10, v10, cape, sp, mslet, pwat (8 vars) |
| **Timesteps** | 81 per variable |

### 2.2 System Resources (Test Machine)

```
Total RAM:      7.8 GB (8,129,720 KB)
Available RAM:  3.7-5.1 GB (varies)
CPU Cores:      2
```

### 2.3 Observed Memory Usage Per Process

| Metric | Value | Notes |
|--------|-------|-------|
| **Peak RSS** | ~860 MB | During variable stacking |
| **Typical RSS** | 650-750 MB | Active processing |
| **Post-GC RSS** | ~550 MB | After variable completion |
| **VmPeak** | ~578 MB | Virtual memory peak |

#### Memory Usage Timeline (10-second sample)

```
Sample  1: 718 MB  (processing)
Sample  2: 750 MB  (processing)
Sample  3: 784 MB  (processing)
Sample  4: 824 MB  (peak - stacking)
Sample  5: 857 MB  (peak)
Sample  6: 543 MB  (GC triggered)
Sample  7: 593 MB  (new variable)
Sample  8: 626 MB  (processing)
Sample  9: 650 MB  (processing)
Sample 10: 683 MB  (processing)
```

**Key Insight**: Memory fluctuates significantly due to:
1. Loading GRIB data into memory (~4 MB per chunk × 81 chunks)
2. Stacking into 3D array
3. Garbage collection between variables

### 2.4 Processing Time Per Process

| Metric | Value |
|--------|-------|
| **Total time (8 vars)** | ~28 seconds |
| **Average per chunk** | ~293 ms |
| **Decoder success rate** | 100% gribberish |

**Note**: Decode time includes network I/O from S3, which is ~80% of the time.

---

## 3. Batch Processing Capacity Calculation

### 3.1 Formula

```python
# Safe available memory (use 80% of available)
safe_memory_mb = available_ram_mb * 0.80

# Memory-based worker limit
mem_workers = safe_memory_mb // peak_memory_per_worker

# CPU-based limit (allow 2x oversubscription for I/O-bound)
cpu_workers = cpu_count * 2

# Optimal workers
optimal = min(mem_workers, cpu_workers)
```

### 3.2 Example Calculation (Test Machine)

```
Available RAM:        5,127 MB
Safe memory (80%):    4,101 MB
Peak per worker:        900 MB (with safety margin)

Memory-based limit:   4,101 / 900 = 4 workers
CPU-based limit:      2 * 2 = 4 workers

>>> RECOMMENDED: 4 parallel workers <<<
```

### 3.3 Scaling Guidelines

| Available RAM | Recommended Workers | Notes |
|---------------|--------------------:|-------|
| 4 GB | 2-3 | Conservative |
| 8 GB | 4-5 | Standard |
| 16 GB | 8-10 | Comfortable |
| 32 GB | 15-20 | Network likely bottleneck |
| 64+ GB | 20-30 | Diminishing returns (I/O bound) |

---

## 4. Using the Batch Processing Script

### 4.1 Profile-Only Mode

Check system capacity without processing:

```bash
python run_batch_gefs_gribberish.py 20250918 00 --profile-only
```

Output:
```
1. System Resource Profile
----------------------------------------
  Total RAM: 7,939 MB (7.8 GB)
  Available RAM: 5,127 MB (5.0 GB)
  RAM Usage: 35.3%
  CPU Cores: 2
  CPU Usage: 12.5%

2. Batch Processing Capacity
----------------------------------------
  Peak memory per worker: ~900 MB (observed)
  Recommended parallel workers: 4
```

### 4.2 Dry Run

See what would be processed:

```bash
python run_batch_gefs_gribberish.py 20250918 00 --members 1-30 --dry-run
```

### 4.3 Full Batch Processing

```bash
# Auto-detect parallelism
python run_batch_gefs_gribberish.py 20250918 00 --members 1-30

# Specify workers manually
python run_batch_gefs_gribberish.py 20250918 00 --members 1-30 --workers 3

# Custom variables
python run_batch_gefs_gribberish.py 20250918 00 --members 1-10 --variables t2m,tp
```

### 4.4 Monitor During Processing

In another terminal:

```bash
# Watch memory usage
watch -n 2 'free -h; echo "---"; ps aux | grep gribberish | grep -v grep'

# Monitor with htop
htop -p $(pgrep -d, -f gribberish)
```

---

## 5. Performance Optimization Tips

### 5.1 Memory Optimization

1. **Process fewer variables at once**:
   ```bash
   # Instead of all 8 variables
   --variables t2m,tp    # Only 2 variables, ~50% less memory
   ```

2. **Use regional subsetting**:
   ```bash
   --region east_africa  # 98% size reduction vs global
   ```

3. **Increase GC frequency** (modify script if needed):
   ```python
   import gc
   gc.collect()  # After each variable
   ```

### 5.2 Throughput Optimization

1. **Network is often the bottleneck** - more workers help with I/O wait
2. **SSD/fast storage** helps with zarr writes
3. **Pre-warm S3 connections** - first request is slower

### 5.3 Recommended Settings by Use Case

| Use Case | Workers | Variables | Region |
|----------|--------:|-----------|--------|
| Quick test | 1 | t2m | east_africa |
| Development | 2 | t2m,tp | east_africa |
| Production (8GB) | 3-4 | all 8 | east_africa |
| Production (16GB+) | 6-8 | all 8 | east_africa |
| Global processing | 2-3 | all 8 | global |

---

## 6. Troubleshooting

### 6.1 Out of Memory

**Symptoms**: Process killed, `MemoryError`, system slowdown

**Solutions**:
1. Reduce `--workers`
2. Reduce `--variables`
3. Process in smaller batches
4. Add swap space (not recommended for performance)

### 6.2 Slow Processing

**Symptoms**: >1s per chunk

**Causes**:
1. Network latency to S3
2. Too many workers competing for I/O
3. CPU throttling

**Solutions**:
1. Check network: `curl -o /dev/null -w "%{time_total}\n" https://noaa-gefs-pds.s3.amazonaws.com/`
2. Reduce workers if CPU-bound
3. Use regional data centers closer to S3

### 6.3 Gribberish Failures

**Symptoms**: Some chunks fail, fallback to cfgrib

**This is expected** for ~1% of chunks with CCSDS compression edge cases.
The hybrid approach handles this automatically.

---

## 7. Example Profiling Session

```bash
# 1. Start a single process in background
python run_single_gefs_to_zarr_gribberish.py 20250918 00 gep01 \
    --region east_africa --variables t2m,tp &

# 2. Get PID
PID=$!
echo "Process PID: $PID"

# 3. Monitor memory every 2 seconds
while kill -0 $PID 2>/dev/null; do
    RSS=$(ps -p $PID -o rss --no-headers)
    echo "$(date +%H:%M:%S) RSS: $((RSS/1024)) MB"
    sleep 2
done

# 4. Analyze peak memory from output
# Peak was ~860 MB → use 900 MB with safety margin
```

---

## 8. Summary

| Metric | Value |
|--------|-------|
| **Peak memory per process** | ~900 MB (8 vars, East Africa) |
| **Recommended workers** | `available_ram_mb * 0.8 / 900` |
| **Processing time** | ~28s per member (8 vars) |
| **Decoder** | 100% gribberish success rate |
| **Throughput** | ~2 members/min (single worker) |

---

*Document created: 2025-12-08*
*Based on profiling of GEFS 20250918 00Z data*
