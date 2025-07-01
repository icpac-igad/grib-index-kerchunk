#!/usr/bin/env python3
"""
Demonstration of GEFS Multi-Ensemble Datatree Structure
This script shows how to structure multiple ensemble members in a single parquet file
using a datatree-compatible format.
"""

import pandas as pd
import numpy as np
import xarray as xr
import json
import os
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import fsspec

warnings.filterwarnings('ignore')

# Configuration
TARGET_DATE_STR = '20250618'
TARGET_RUN = '00'
EA_LAT_MIN, EA_LAT_MAX = -12, 15
EA_LON_MIN, EA_LON_MAX = 25, 52

print("🎯 GEFS Ensemble Datatree Structure Demonstration")
print("="*60)


def read_parquet_fixed(parquet_path):
    """Read parquet files with proper handling."""
    import ast
    
    df = pd.read_parquet(parquet_path)
    
    if 'refs' in df['key'].values and len(df) <= 2:
        refs_row = df[df['key'] == 'refs']
        refs_value = refs_row['value'].iloc[0]
        if isinstance(refs_value, bytes):
            refs_value = refs_value.decode('utf-8')
        zstore = ast.literal_eval(refs_value)
    else:
        zstore = {}
        for _, row in df.iterrows():
            key = row['key']
            value = row['value']
            
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            
            if isinstance(value, str):
                if value.startswith('[') and value.endswith(']'):
                    try:
                        value = json.loads(value)
                    except:
                        pass
            
            zstore[key] = value
    
    if 'version' in zstore:
        del zstore['version']
    
    return zstore


def create_ensemble_datatree_parquet(base_parquet='gefs_gep01_20250618_00z_fixed.par'):
    """
    Create a demonstration ensemble datatree parquet by duplicating gep01 data
    to simulate multiple ensemble members.
    """
    print("\n📊 Creating ensemble datatree structure...")
    
    # Read the base gep01 parquet
    print(f"📖 Reading base parquet: {base_parquet}")
    base_store = read_parquet_fixed(base_parquet)
    
    # Create ensemble members by simulating variations
    ensemble_members = ['gep01', 'gep02', 'gep03', 'gep04', 'gep05']
    
    # Create combined store with proper datatree structure
    combined_store = {}
    
    # Add root metadata
    root_attrs = {
        "Conventions": "CF-1.6",
        "GRIB_edition": 2,
        "GRIB_centre": "kwbc",
        "history": f"GEFS ensemble forecast with {len(ensemble_members)} members",
        "ensemble_members": ensemble_members,
        "institution": "NOAA/NCEP",
        "source": "GEFS forecast system"
    }
    combined_store['.zattrs'] = json.dumps(root_attrs)
    
    # Add each ensemble member
    for i, member in enumerate(ensemble_members):
        print(f"\n📦 Processing {member}...")
        
        # For demonstration, we'll use the same data but could apply perturbations
        member_prefix = f"{member}/"
        
        # Copy all entries with member prefix
        entries_added = 0
        for key, value in base_store.items():
            # Skip version key
            if key == 'version':
                continue
            
            # Add member prefix to create hierarchy
            new_key = member_prefix + key
            
            # For demonstration, slightly modify zarr references for different members
            if isinstance(value, list) and len(value) == 3 and i > 0:
                # This would be where ensemble perturbations are reflected
                # For now, just use the same references
                pass
            
            combined_store[new_key] = value
            entries_added += 1
        
        print(f"   ✅ Added {entries_added} entries for {member}")
    
    # Save combined parquet
    output_file = f'gefs_ensemble_demo_{TARGET_DATE_STR}_{TARGET_RUN}z.par'
    print(f"\n💾 Saving ensemble parquet: {output_file}")
    
    # Convert to DataFrame format
    data = []
    for key, value in combined_store.items():
        if isinstance(value, str):
            encoded_value = value.encode('utf-8')
        elif isinstance(value, (list, dict)):
            encoded_value = json.dumps(value).encode('utf-8')
        else:
            encoded_value = str(value).encode('utf-8')
        
        data.append((key, encoded_value))
    
    df = pd.DataFrame(data, columns=['key', 'value'])
    df.to_parquet(output_file)
    
    print(f"✅ Saved ensemble parquet with {len(df)} rows")
    print(f"   - {len(ensemble_members)} ensemble members")
    print(f"   - {len(df) // len(ensemble_members)} entries per member (approx)")
    
    return output_file


def read_ensemble_datatree(parquet_path):
    """
    Read and demonstrate accessing ensemble datatree structure.
    """
    print(f"\n📖 Reading ensemble datatree from {parquet_path}...")
    
    # Read parquet
    zstore = read_parquet_fixed(parquet_path)
    
    # Analyze structure
    print("\n🔍 Analyzing datatree structure:")
    
    # Find ensemble members
    members = set()
    for key in zstore.keys():
        if '/' in key and not key.startswith('.'):
            member = key.split('/')[0]
            if member.startswith('gep'):
                members.add(member)
    
    members = sorted(members)
    print(f"   Found {len(members)} ensemble members: {', '.join(members)}")
    
    # Count entries per member
    for member in members:
        member_keys = [k for k in zstore.keys() if k.startswith(f"{member}/")]
        print(f"   - {member}: {len(member_keys)} entries")
    
    # Create reference filesystem
    print("\n🌐 Creating reference filesystem...")
    fs = fsspec.filesystem("reference", fo=zstore, remote_protocol='s3', 
                          remote_options={'anon': True})
    mapper = fs.get_mapper("")
    
    # Open as datatree
    print("📂 Opening as datatree...")
    dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)
    
    # Show structure
    print("\n📊 Datatree structure:")
    print(f"   Root groups: {list(dt.groups)}")
    
    # Access specific member and variable
    if 'gep01' in dt.groups:
        print(f"\n🔍 Exploring gep01 structure:")
        gep01_tree = dt['gep01']
        
        # Find available variables
        for path in gep01_tree.groups:
            print(f"   - {path}")
            if hasattr(gep01_tree[path], 'ds'):
                vars_list = list(gep01_tree[path].ds.data_vars)
                if vars_list:
                    print(f"     Variables: {vars_list}")
    
    return dt


def plot_ensemble_from_datatree(dt, variable='tp', timestep=8):
    """
    Create a plot from the ensemble datatree.
    """
    print(f"\n🎨 Plotting ensemble {variable} at timestep {timestep} (T+{timestep*3}h)...")
    
    # Get ensemble members
    members = [g for g in dt.groups if g.startswith('gep')]
    members.sort()
    
    num_members = len(members)
    if num_members == 0:
        print("❌ No ensemble members found")
        return
    
    # Create figure
    cols = 3
    rows = (num_members + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes = axes.flatten()
    
    # Plot each member
    for idx, member in enumerate(members):
        ax = axes[idx]
        
        try:
            # Access data
            if variable == 'tp':
                data = dt[f'/{member}/tp/accum/surface'].ds['tp']
            else:
                print(f"Variable {variable} not implemented")
                continue
            
            # Extract East Africa
            ea_data = data.sel(
                latitude=slice(EA_LAT_MAX, EA_LAT_MIN),
                longitude=slice(EA_LON_MIN, EA_LON_MAX)
            ).isel(valid_times=timestep)
            
            # Compute values
            values = ea_data.compute().values
            
            # Plot
            levels = np.linspace(0, 30, 11)
            im = ax.contourf(ea_data.longitude, ea_data.latitude, values,
                           levels=levels, cmap='Blues',
                           transform=ccrs.PlateCarree(),
                           extend='max')
            
            # Add features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.LAND, alpha=0.1)
            
            ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX])
            
            gl = ax.gridlines(draw_labels=True, alpha=0.3)
            gl.top_labels = False
            gl.right_labels = False
            
            ax.set_title(f'{member} - T+{timestep*3}h')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'{member}\nError', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{member} - Error')
            print(f"   ⚠️ Error plotting {member}: {e}")
    
    # Hide unused subplots
    for idx in range(num_members, len(axes)):
        axes[idx].set_visible(False)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Total Precipitation (mm)', rotation=270, labelpad=20)
    
    # Title
    plt.suptitle(f'GEFS Ensemble {variable.upper()} - Datatree Demo\n'
                f'{TARGET_DATE_STR} {TARGET_RUN}Z Run - T+{timestep*3}h',
                fontsize=16, y=0.98)
    
    # Save
    output_file = f'gefs_ensemble_datatree_demo_{variable}.png'
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot: {output_file}")
    plt.close()
    
    return output_file


def main():
    """Main demonstration function."""
    print("\n" + "="*60)
    print("GEFS Ensemble Datatree Demonstration")
    print("="*60)
    
    # Check if we have gep01 parquet
    base_parquet = f'gefs_gep01_{TARGET_DATE_STR}_{TARGET_RUN}z_fixed.par'
    if not os.path.exists(base_parquet):
        print(f"❌ Base parquet not found: {base_parquet}")
        print("   Please run the single member processing first.")
        return False
    
    # Create ensemble datatree parquet
    ensemble_parquet = create_ensemble_datatree_parquet(base_parquet)
    
    # Read and analyze the datatree
    try:
        dt = read_ensemble_datatree(ensemble_parquet)
        
        # Create visualization
        plot_file = plot_ensemble_from_datatree(dt, 'tp', timestep=8)
        
        print("\n✅ Demonstration completed successfully!")
        print(f"   - Ensemble parquet: {ensemble_parquet}")
        print(f"   - Visualization: {plot_file}")
        
        # Show how to access data programmatically
        print("\n📝 Example code to access ensemble data:")
        print("```python")
        print("# Read parquet")
        print("zstore = read_parquet_fixed('gefs_ensemble_demo_20250618_00z.par')")
        print("")
        print("# Create reference filesystem")
        print("fs = fsspec.filesystem('reference', fo=zstore, remote_protocol='s3',")
        print("                      remote_options={'anon': True})")
        print("mapper = fs.get_mapper('')")
        print("")
        print("# Open as datatree")
        print("dt = xr.open_datatree(mapper, engine='zarr', consolidated=False)")
        print("")
        print("# Access specific member and variable")
        print("tp_gep01 = dt['/gep01/tp/accum/surface'].ds['tp']")
        print("tp_gep02 = dt['/gep02/tp/accum/surface'].ds['tp']")
        print("")
        print("# Extract region and compute")
        print("ea_data = tp_gep01.sel(latitude=slice(15, -12),")
        print("                       longitude=slice(25, 52))")
        print("values = ea_data.compute().values")
        print("```")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Datatree demonstration completed!")
    else:
        print("\n❌ Demonstration failed.")
        exit(1)