import pandas as pd
import datatree
import fsspec
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import gc 

def read_and_process_parquet(file_path):
    """
    Read a Parquet file, decode byte strings in the 'value' column, and convert to a dictionary.

    Parameters:
    file_path (str): The path to the Parquet file to read.

    Returns:
    dict: A dictionary with keys and values extracted from the Parquet file.
    """
    df = pd.read_parquet(file_path)
    df['value'] = df['value'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    return {row['key']: eval(row['value']) for _, row in df.iterrows()}

def load_data_from_dict(zstore, var_path):
    """
    Load data into a DataTree using a dictionary source and return a subselected xarray dataset based on a specific variable path.

    Parameters:
    zstore (dict): Dictionary containing zarr store information.
    var_path (str): The specific variable path within the data structure to be loaded.

    Returns:
    xarray.Dataset: The dataset loaded from the DataTree, subselected by a predefined latitude and longitude range.
    """
    gfs_dt = datatree.open_datatree(fsspec.filesystem("reference", fo=zstore).get_mapper(""), engine="zarr", consolidated=False)
    aa = gfs_dt[var_path].to_dataset()
    aa1 = aa.drop_vars('step')
    aa2 = aa1.compute()

    # Example latitude and longitude range for subselection
    lat_min, lat_max = 1, 5
    lon_min, lon_max = 25, 30
    return aa2.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

def plot_data(data, var_name, file_date):
    """
    Plot subselected data on a map with multiple subplots and save the plot with a variable-specific name.

    Parameters:
    data (xarray.Dataset): The dataset containing the data to plot.
    var_name (str): The variable name used for saving the plot file.
    file_date (str): Date extracted from the file name for use in the plot file name.
    """
    fig, axes = plt.subplots(nrows=16, ncols=8, figsize=(40, 80), subplot_kw={'projection': ccrs.PlateCarree()}, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    axes = axes.flatten()

    for idx, dstime in enumerate(data.valid_times):
        ax = axes[idx]
        pdata = data.tp.sel(valid_times=dstime)
        input_str = pdata['valid_time'].values
        pdata.plot(ax=ax, add_colorbar=False, transform=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_title(str(input_str)[:16], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('lon')
        ax.set_ylabel('lat')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.2)
    # Generate a filename using the variable name and file date
    filename = f"{var_name}-{file_date}.png"
    plt.savefig(filename, dpi=200)
    plt.show()
    plt.close(fig)  # Explicitly close the figure to free memory
    del data  # Delete the data to free memory
    gc.collect()  # Force garbage collection

def extract_date_from_filename(file_path):
    """
    Extract the date from a file path.

    Parameters:
    file_path (str): The file path from which to extract the date.

    Returns:
    str: The extracted date as a string.
    """
    base_name = os.path.basename(file_path)
    date_str = base_name.split('-')[3]  # Assuming the date is always in the same position
    return date_str



list_ds=['20210501','20210803','20211108','20220104','20220501','20220815',
         '20221201','20230108','20240405','20240928','20241001']

for date_str in list_ds:
    parquet_file_path = f'gfs-z00-{date_str}-allvar.parquet'
    variable_path = "tp/accum/surface"
    var_name = variable_path.split('/')[0]

    zstore_dict = read_and_process_parquet(parquet_file_path)
    data = load_data_from_dict(zstore_dict, variable_path)
    
    #file_date = extract_date_from_filename(parquet_file_path)
    file_date=date_str
    plot_data(data, var_name, file_date)
    
    del zstore_dict, data  # Free memory by deleting large variables
    gc.collect()  # Run garbage collector after the loop iteration
    print(f'completed {date_str}')
