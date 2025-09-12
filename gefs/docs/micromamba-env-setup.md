# Micromamba Environment Setup for GEFS Processing

## Overview
This guide provides comprehensive instructions for setting up the GEFS processing environment using Micromamba across different platforms (Linux, Windows) and also includes Coiled cloud environment setup.

## Table of Contents
1. [Linux Environment Setup](#linux-environment-setup)
2. [Windows Environment Setup](#windows-environment-setup)
3. [Coiled Cloud Environment Setup](#coiled-cloud-environment-setup)

---

## Linux Environment Setup

### Prerequisites
- Linux system (Ubuntu, CentOS, RHEL, etc.)
- Internet connection
- Bash shell

### Step 1: Install Micromamba on Linux

#### Automated Installation (Recommended)
```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc
micromamba self-update -c conda-forge
```

#### Manual Installation Alternative
```bash
# Download micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

# Move to system location
sudo mv bin/micromamba /usr/local/bin/

# Initialize micromamba
micromamba shell init --shell bash
source ~/.bashrc
```

### Step 2: Create Environment from YAML File

The repository contains two YAML files for different use cases:

#### Option A: Complete Environment (vz_gik_env.yaml)
This is the comprehensive environment file with all dependencies:

```bash
# Navigate to the GEFS project directory
cd /path/to/gefs

# Create environment from the main YAML file
micromamba create --name gik-zarrv2 --file vz_gik_env.yaml

# Activate the environment
micromamba activate gik-zarrv2
```

#### Option B: Coiled-Compatible Environment (env/vz-dask-zarr2.yaml)
This is optimized for cloud processing with Coiled:

```bash
# Create environment from the Coiled-compatible YAML file
micromamba create --name geospatial-data-env --file env/vz-dask-zarr2.yaml

# Activate the environment
micromamba activate geospatial-data-env
```

### Step 3: Verify Linux Installation
```bash
# Check Python version
python --version

# List installed packages
micromamba list

# Test key imports
python -c "import xarray, zarr, kerchunk, dask; print('All key packages imported successfully')"
```

---

## Windows Environment Setup

### Prerequisites
- Windows 10 or later
- PowerShell (recommended) or Command Prompt
- Internet connection

### Step 1: Install Micromamba on Windows

#### Option A: Using PowerShell (Recommended)
Open PowerShell as Administrator and run:

```powershell
Invoke-Webrequest -URI https://micro.mamba.pm/api/micromamba/win-64/latest -OutFile micromamba.tar.bz2
tar xf micromamba.tar.bz2
```

Or use the automated installer:
```powershell
Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1).Content)
```

#### Option B: Manual Installation
1. Download the latest Windows installer from: https://micro.mamba.pm/install.html
2. Extract the downloaded file to a suitable location (e.g., `C:\micromamba`)
3. Add the micromamba directory to your system PATH environment variable

### Step 2: Initialize Micromamba
After installation, open a new PowerShell window and initialize micromamba:

```powershell
micromamba shell init --shell powershell
```

Close and reopen PowerShell to apply the changes.

### Step 3: Update Micromamba
```powershell
micromamba self-update -c conda-forge
```

### Step 4: Create the GEFS Environment from YAML Files
Navigate to the directory containing the YAML files and create the environment:

#### Using the main environment file:
```powershell
micromamba create --name gik-zarrv2 --file vz_gik_env.yaml
```

#### Using the Coiled-compatible environment file:
```powershell
micromamba create --name geospatial-data-env --file env/vz-dask-zarr2.yaml
```

### Step 5: Activate the Environment
```powershell
micromamba activate gik-zarrv2
# OR
micromamba activate geospatial-data-env
```

### Windows Troubleshooting

#### 1. PowerShell Execution Policy
If you encounter execution policy errors, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 2. PATH Issues
If micromamba is not recognized after installation:
1. Check if the micromamba directory is in your PATH
2. Manually add it to your system PATH environment variable
3. Restart PowerShell/Command Prompt

#### 3. SSL Certificate Errors
If you encounter SSL certificate errors:
```powershell
micromamba config --set ssl_verify false
```

#### 4. Alternative: Using Command Prompt
If you prefer using Command Prompt instead of PowerShell:

1. Open Command Prompt as Administrator
2. Follow the same installation steps, but use `cmd` instead of `powershell` in the shell init command:
   ```cmd
   micromamba shell init --shell cmd.exe
   ```

---

## Coiled Cloud Environment Setup

### Overview
Coiled provides cloud-based Dask clusters for scalable processing. This setup is ideal for large-scale GEFS data processing.

### Prerequisites
- Coiled account (sign up at https://coiled.io)
- Google Cloud Platform account (for GCS access)
- Service account credentials

### Step 1: Install Coiled
If not already included in your micromamba environment:

```bash
# Linux/macOS
micromamba install -c conda-forge coiled

# Windows
micromamba install -c conda-forge coiled
```

### Step 2: Configure Coiled Authentication
```bash
# Login to your Coiled account
coiled login

# Set up your workspace (replace with your workspace name)
coiled config set workspace geosfm
```

### Step 3: Create Coiled Software Environment
Create a software environment using the YAML file:

```bash
# Create a Coiled software environment from the YAML file
coiled software create --name gik-zarr2 --file env/vz-dask-zarr2.yaml
```

### Step 4: Start Coiled Notebook/Cluster
```bash
# Start a Coiled notebook instance
coiled notebook start --name dask-thresholds --vm-type n2-standard-2 --software gik-zarr2 --workspace=geosfm

# Or start a Dask cluster
coiled cluster start --name gefs-processing --vm-type n2-standard-4 --software gik-zarr2 --workers 4
```

### Step 5: Upload Required Files
Upload these essential files to your Coiled workspace:
1. `gefs_util.py` - Utility functions
2. `run_day_gefs_ensemble_full.py` - Main processing script
3. `run_gefs_24h_accumulation.py` - Accumulation processing
4. `ea_ghcf_simple.geojson` - Geographic boundaries
5. Service account JSON file for GCP authentication

### Step 6: Set Up Authentication
```python
# In your Coiled notebook/cluster
import os
from google.cloud import storage

# Set up GCP service account (upload your service account JSON)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/path/to/service-account.json'

# Test GCS access
client = storage.Client()
buckets = list(client.list_buckets())
print(f"Accessible buckets: {[b.name for b in buckets]}")
```

## Verification Steps

### For All Environments
After setting up any environment, verify the installation:

```python
# Test script to verify environment
import sys
print(f"Python version: {sys.version}")

# Test core packages
packages_to_test = [
    'xarray', 'zarr', 'kerchunk', 'dask', 'distributed', 
    'pandas', 'numpy', 'matplotlib', 'fsspec', 's3fs', 
    'gcsfs', 'pyarrow', 'cartopy', 'geopandas'
]

for package in packages_to_test:
    try:
        __import__(package)
        print(f"✓ {package}")
    except ImportError:
        print(f"✗ {package} - NOT FOUND")
```

## YAML File Comparison

### vz_gik_env.yaml (Main Environment)
- **Purpose**: Complete local development environment
- **Size**: 130+ packages with specific versions
- **Use case**: Local development, testing, full feature set

### env/vz-dask-zarr2.yaml (Coiled Environment)  
- **Purpose**: Cloud-optimized environment for Coiled
- **Size**: ~80 packages with flexible versions
- **Use case**: Cloud processing, Coiled clusters, production workflows

## Additional Resources
- Official Micromamba documentation: https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html
- Coiled documentation: https://docs.coiled.io/
- Windows-specific installation guide: https://mamba.readthedocs.io/en/latest/installation.html#windows
- Troubleshooting guide: https://mamba.readthedocs.io/en/latest/user_guide/troubleshooting.html

## Notes
- Always activate the appropriate environment before running GEFS processing scripts
- For Windows users, consider using WSL2 (Windows Subsystem for Linux) for better compatibility
- The Coiled setup requires proper GCP authentication for accessing GCS buckets
- Choose the appropriate YAML file based on your use case (local vs cloud processing)