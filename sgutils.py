# Python standard libraries
import os
import sys
import re
import calendar
import json
import tempfile
from datetime import datetime, timedelta
import ntpath

# Third-party libraries
import numpy as np
import pandas as pd
import xarray as xr
import ujson
import fsspec
import dask
import dask.bag as daskbag
import dask.dataframe as daskdf
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import geopandas as gp
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import ImageGrid
import coiled
import kerchunk
from kerchunk.grib2 import scan_grib
from kerchunk.combine import MultiZarrToZarr
from scipy.ndimage import zoom

# %%01-delayed-get-gefs-kc-daily


def flatten_list(list_of_lists):
    flattened_list = []
    for sublist in list_of_lists:
        flattened_list.extend(sublist)
    return flattened_list


def get_details(url):
    pattern = r"s3://noaa-gefs-pds/gefs\.(\d+)/(\d+)/atmos/pgrb2sp25/gep(\w+)\.t(\d+)z\.pgrb2s\.0p25.f(\d+)"
    match = re.match(pattern, url)
    if match:
        date = match.group(1)
        run = match.group(2)
        ens_mem = match.group(3)
        hour = match.group(4)
        return date, run, hour, ens_mem
    else:
        print("No match found.")
        return None


def foldercreator(path):
    """
    creates a folder

    Parameters
    ----------
    path : folder path

    Returns
    -------
    creates a folder
    """
    if not os.path.exists(path):
        os.makedirs(path)


@dask.delayed
def gen_json(s3_url):
    s3_source = {"anon": True, "skip_instance_cache": True}
    var_filter = {"typeOfLevel": "surface", "name": "Total Precipitation"}
    date, run, hour, ens_mem = get_details(s3_url)
    year = date[:4]
    month = date[4:6]
    # fs = fsspec.filesystem("s3", aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    fs = fsspec.filesystem("s3")
    max_retry = 5  # Number of maximum retries
    while max_retry >= 0:
        try:
            out = scan_grib(s3_url, storage_options=s3_source, filter=var_filter)[0]
            # print('scan grib is ok')
            flname = s3_url.split("/")[-1]
            output_flname = f"s3://arco-ibf/fcst/gefs_ens/{year}/{month}/{date}/{run}/individual/{flname}.json"
            with fs.open(output_flname, "w") as f:
                f.write(ujson.dumps(out))
            break
        except Exception:
            if max_retry == 0:
                # If the maximum number of retries has been reached, raise the exception
                raise
            else:
                # If an exception occurs, decrement the retry counter
                max_retry -= 1
                print(f"Retrying... Remaining retries: {max_retry+1}")
    return output_flname


def gefs_s3_utl_maker(date, run):
    fs_s3 = fsspec.filesystem("s3", anon=True)
    members = [str(i).zfill(2) for i in range(1, 31)]
    s3url_ll = []
    for ensemble_member in members:
        s3url_glob = fs_s3.glob(
            f"s3://noaa-gefs-pds/gefs.{date}/{run}/atmos/pgrb2sp25/gep{ensemble_member}.*"
        )
        s3url_only_grib = [f for f in s3url_glob if f.split(".")[-1] != "idx"]
        fmt_s3og = sorted(["s3://" + f for f in s3url_only_grib])
        s3url_ll.append(fmt_s3og[1:])
    gefs_url = [item for sublist in s3url_ll for item in sublist]
    return gefs_url


def xcluster_process_kc_individual(date, run):
    gefs_url = gefs_s3_utl_maker(date, run)
    print(len(gefs_url))

    cluster = coiled.Cluster(
        n_workers=5,
        name=f"gks1-{date}-{run}",
        software="v3-gefs-run-x64-20231113",
        #workspace='argee-sn',
        # scheduler_cpu=2,
        # scheduler_memory="2 GiB",
        scheduler_vm_types=["t3.small", "t3a.small"],
        region="us-east-1",
        arm=False,
        compute_purchase_option="spot",
        tags={"workload": "gefs-arm-test0"},
        worker_vm_types="t3.small",
    )

    client = cluster.get_client()
    client.upload_file("utils.py")
    results = []
    for input_value in gefs_url:
        result = gen_json(input_value)
        results.append(result)
    final_results = dask.compute(results)

    client.close()
    cluster.shutdown()

    return final_results


# %%02-combine-gefs-json-func


def foldercreator(path):
    """
    creates a folder

    Parameters
    ----------
    path : folder path

    Returns
    -------
    creates a folder
    """
    if not os.path.exists(path):
        os.makedirs(path)


def path_leaf(path):
    """
    Get the name of a file without any extension from given path

    Parameters
    ----------
    path : file full path with extension

    Returns
    -------
    str
       filename in the path without extension

    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def grib_flname_get_details(url):
    pattern = r"gep(\w+)\.t(\d+)z\.pgrb2s\.0p25.f(\d+)"
    match = re.match(pattern, url)
    if match:
        ens_mem = match.group(1)
        run = match.group(2)
        hour = match.group(3)
        return ens_mem, run, hour
    else:
        print("No match found.")
        return None


def gefs_mem_list(lj_glob, date, run, member):
    year = date[:4]
    month = date[4:6]
    flnames = [path_leaf(i) for i in lj_glob]
    df = pd.DataFrame()
    df["flnames"] = flnames
    df["ens_mem"], df["run"], df["hour"] = zip(
        *df["flnames"].apply(lambda x: grib_flname_get_details(x))
    )
    db = df[df["ens_mem"] == member]
    db1 = db.sort_values("hour")
    db1["flname"] = db1.apply(
        lambda row: f"gep{row['ens_mem']}.t{row['run']}z.pgrb2s.0p25.f{row['hour']}.json",
        axis=1,
    )
    fn_list = db1["flname"].tolist()
    s3_url_fn_list = [
        f"s3://arco-ibf/fcst/gefs_ens/{year}/{month}/{date}/{run}/individual/" + f
        for f in fn_list
    ]
    return s3_url_fn_list


@coiled.function(
    # memory="2 GiB",
    vm_type="t3.small",
    software="v3-gefs-run-x64-20231113",
    name=f"func-combine-gefs",
    region="us-east-1",  # Specific region
    arm=False,  # Change architecture
    idle_timeout="25 minutes",
)
def combine(s3_json_urls, date, run):
    with tempfile.TemporaryDirectory() as temp_dir:
        local_paths = []
        # Download files to temporary directory
        for url in s3_json_urls:
            localfl = path_leaf(url)
            print(localfl)
            local_path = os.path.join(temp_dir, os.path.basename(localfl))
            print(local_path)
            print(url)
            fs_s3 = fsspec.filesystem("s3", anon=False)
            with fs_s3.open(url, "rb") as remote_file:
                with open(local_path, "wb") as local_file:
                    local_file.write(remote_file.read())
            local_paths.append(local_path)
        mzz = MultiZarrToZarr(local_paths, concat_dims=["valid_time"])
        flname = (
            f"{'.'.join(s3_json_urls[0].split('/')[-1].split('.')[:-2])}.combined.json"
        )
        local_output_flname = os.path.join(temp_dir, flname)
        print(local_output_flname)
        fs_local = fsspec.filesystem("")
        with fs_local.open(local_output_flname, "w") as f:
            f.write(ujson.dumps(mzz.translate()))
        year = date[:4]
        month = date[4:6]
        s3_output_flname = (
            f"s3://arco-ibf/fcst/gefs_ens/{year}/{month}/{date}/{run}/{flname}"
        )
        fs = fsspec.filesystem("s3")
        with fs.open(s3_output_flname, "w") as f:
            f.write(ujson.dumps(mzz.translate()))
    return "done for a memeber"


def func_combine_kc(date, run):
    year = date[:4]
    month = date[4:6]
    fs_s3 = fsspec.filesystem("s3", anon=False)
    lj_glob = fs_s3.glob(
        f"s3://arco-ibf/fcst/gefs_ens/{year}/{month}/{date}/{run}/individual/*.json"
    )
    members = [str(i).zfill(2) for i in range(1, 31)]

    combine.cluster.adapt(minimum=8, maximum=10)
    func_env = combine.cluster.get_client()
    func_env.upload_file("utils.py")
    futures = []

    for member in members:
        s3_json_urls = gefs_mem_list(lj_glob, date, run, member)
        result = combine.submit(s3_json_urls, date, run)
        futures.append(result)

    return [f.result() for f in futures]


# %%03-dask-delay-plot-coiled-arm


def make_kc_zarr_df(date, run):
    fs_s3 = fsspec.filesystem("s3", anon=False)
    # combined = fs_s3.glob(f"s3://arco-ibf/fcst/gefs_ens/{date}/{run}/gep*")
    year = date[:4]
    month = date[4:6]
    combined = fs_s3.glob(
        f"s3://arco-ibf/fcst/gefs_ens/{year}/{month}/{date}/{run}/gep*"
    )
    combined1 = ["s3://" + f for f in combined]
    mzz = MultiZarrToZarr(
        combined1,
        remote_protocol="s3",
        remote_options={"anon": False},
        concat_dims=["number"],
        identical_dims=["valid_time", "longitude", "latitude"],
    )
    out = mzz.translate()
    fs_ = fsspec.filesystem(
        "reference", fo=out, remote_protocol="s3", remote_options={"anon": True}
    )
    m = fs_.get_mapper("")
    ds = xr.open_dataset(m, engine="zarr", backend_kwargs=dict(consolidated=False))
    return ds


def create_dataframe_from_s3(bucket_name, folder_location, retry_atmpt):
    # Create a session using your AWS credentials

    session = boto3.Session()

    # Create an S3 client using the session
    s3_client = session.client("s3")

    # List objects in the specified bucket and folder
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_location)

    # Extract filenames and file sizes from the response
    filenames = []
    file_sizes = []
    for obj in response["Contents"]:
        filenames.append(obj["Key"].split("/")[-1])
        file_sizes.append(obj["Size"])

    # Create a pandas dataframe with the filenames and file sizes
    df = pd.DataFrame({"Filename": filenames, "Filesize": file_sizes})
    mean_size = df["Filesize"].mean()
    std_size = df["Filesize"].std()
    lower_threshold = mean_size - std_size
    upper_threshold = mean_size + std_size
    if not retry_atmpt == 2:
        filtered_df = df[
            (df["Filesize"] >= lower_threshold) & (df["Filesize"] <= upper_threshold)
        ]
        # df1=df[[~filtered_df]]
        df1 = df[~df["Filename"].isin(filtered_df["Filename"])]
        df1["time"] = df1["Filename"].str.split("_").str[0]
        df1["time1"] = pd.to_datetime(df1["time"], format="%Y%m%d%H")
        # Create a new column 'time2' with the desired format
        df1["time2"] = df1["time1"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        pattern = r"m(\d+)"
        df1["member"] = df1["Filename"].str.extract(pattern)
        df2 = df1[["Filename", "Filesize", "member", "time2"]]
    else:
        df1 = df[df["Filesize"] == 13939]
        df1["time"] = df1["Filename"].str.split("_").str[0]
        df1["time1"] = pd.to_datetime(df1["time"], format="%Y%m%d%H")
        # Create a new column 'time2' with the desired format
        df1["time2"] = df1["time1"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        pattern = r"m(\d+)"
        df1["member"] = df1["Filename"].str.extract(pattern)
        df2 = df1[["Filename", "Filesize", "member", "time2"]]
    return df2


@dask.delayed
def plot(df, time, member, date, run):
    is_six_hourly = time in df.valid_time.values[df.valid_time.dt.hour % 6 == 0]
    if is_six_hourly:
        six_hourly_steps = time
        print(f"checked six hour {six_hourly_steps}")
        prev_3hr_steps = six_hourly_steps - np.timedelta64(3, "h")
        print(f"previous 3 hour step and it is {prev_3hr_steps}")
        prev_3hr_data = df.sel(valid_time=prev_3hr_steps)
        prev_6hr_data = df.sel(valid_time=six_hourly_steps)
        corr_hr = prev_6hr_data - prev_3hr_data
        df = corr_hr.assign_coords(valid_time=[six_hourly_steps])
        print(f"corrected the tp for step {six_hourly_steps}")
    else:
        pass

    # Boundaries from the figure
    boundaries = [
        0,
        0.08,
        0.15,
        0.3,
        0.65,
        1.3,
        2.7,
        5.6,
        11.5,
        23.7,
        48.6,
        100,
        205,
        400,
        864,
        1775,
    ]

    # Colors from the figure, adding white for values below 0.08
    colors = [
        "#FFFFFF",
        "#00FFFF",
        "#0099FF",
        "#3333FF",
        "#00FF00",
        "#009900",
        "#FFCC00",
        "#FF6600",
        "#FF0000",
        "#CC0000",
        "#990000",
        "#660000",
        "#330000",
        "#FF00FF",
        "#993399",
    ]

    colors = [
        "#325d32",
        "#1d7407",
        "#2a9c0a",
        "#3333FF",
        "#00FF00",
        "#009900",
        "#FFCC00",
        "#FF6600",
        "#FF0000",
        "#CC0000",
        "#990000",
        "#660000",
        "#330000",
        "#FF00FF",
        "#993399",
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", colors, N=len(boundaries) - 1
    )

    fig = plt.figure(figsize=(10, 8))
    s3 = fsspec.filesystem("s3")
    json_file = "s3://arco-ibf/vectors/ea_ghcf_simple.json"

    with s3.open(json_file, "r") as f:
        geom = json.load(f)

    gdf = gp.GeoDataFrame.from_features(geom)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_geometries(
        gdf["geometry"], crs=ccrs.PlateCarree(), facecolor="none", edgecolor="black"
    )

    # Use boundaries and cmap in the plot method
    df.sel(valid_time=time)["tp"].plot(
        ax=ax, levels=boundaries, cmap=cmap, extend="both", add_colorbar=False
    )
    # Add text label at the bottom corner
    ax.set_title("")
    ax.text(
        0.95, 0.95, f"m{member}", transform=ax.transAxes, fontsize=12, fontweight="bold"
    )
    date_fmt = pd.to_datetime(str(time))
    year, month = date_fmt.year, date_fmt.month
    month_fmt = str(month).zfill(2)
    ind_date_fmt1 = date_fmt.strftime("%Y%m%d")
    ind_file_date_fmt1 = date_fmt.strftime("%Y%m%d%H")
    img_output = f"{ind_file_date_fmt1}_m{member}.jpg"

    fig.savefig(img_output, dpi=100, bbox_inches="tight")
    plt.close()

    s3_client = boto3.client("s3")
    folder_year = date[:4]
    folder_month = date[4:6]
    folder_date = date
    s3_location = f"fcst/gefs_ens/{folder_year}/{folder_month}/{folder_date}/{run}/plot_individual/{img_output}"

    s3_client.upload_file(img_output, "arco-ibf", s3_location)
    os.remove(img_output)


class acluster_RetryPlotter:
    def __init__(self, date, run, min_lon=21, min_lat=-12, max_lon=53, max_lat=24):
        self.date = date
        self.run = run
        self.cropped_ds = self._crop_dataset()
        self.avlbl_members = len(self.cropped_ds["number"].values)
        self.cluster_name = "cartopy-gefs-plot-arm"

    def _initialize_cluster(self, name):
        return coiled.Cluster(
            n_workers=5,
            name=f"gplot-{self.date}-{self.run}",
            software="v3-gefs-run-arm-20231113",
            # scheduler_memory="8 GiB",
            scheduler_vm_types=["t4g.small", "t4g.medium"],
            region="us-east-1",
            arm=True,
            compute_purchase_option="spot",
            tags={"workload": "gefs-arm-test0"},
            worker_vm_types=["t4g.large"],
        )

    def _crop_dataset(self):
        min_lon = 21
        min_lat = -12
        max_lon = 53
        max_lat = 24
        ds = make_kc_zarr_df(self.date, self.run)
        self.sub_ds = ds.sel(
            latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
        )
        return self.sub_ds

    def _process_tasks(self, tasks):
        client = self.cluster.get_client()
        client.upload_file("utils.py")
        dask.compute(tasks)
        client.close()

    def _get_retry_tasks(self, retry_df):
        tasks = []
        for idx, row in retry_df.iterrows():
            df = self.cropped_ds.isel(number=int(row["member"]))
            member = row["member"]
            time = row["time2"]
            tasks.append(plot(df, time, member, self.date, self.run))
        return tasks

    def run_plotter(self):
        self.cluster = self._initialize_cluster(self.cluster_name)
        tasks = []
        avlbl_em = self.avlbl_members
        for member in np.arange(0, avlbl_em):
            df = self.cropped_ds.isel(number=member)
            for time in df["valid_time"].values[0:16]:
                tasks.append(plot(df, time, member, self.date, self.run))

        self._process_tasks(tasks)

    def retry(self, attempt=1):
        self.cluster = self._initialize_cluster(self.cluster_name)
        bucket_name = "arco-ibf"
        folder_location = f"fcst/gefs_ens/{self.date[:4]}/{self.date[4:6]}/{self.date}/{self.run}/plot_individual/"
        retry_df = create_dataframe_from_s3(bucket_name, folder_location, attempt)

        if not retry_df.empty:
            # retry_df.to_csv(f'{self.date}_{self.run}_rt{attempt}.csv')
            tasks = self._get_retry_tasks(retry_df)
            self._process_tasks(tasks)

    def shutdown(self):
        self.cluster.shutdown()


# %%04-plot-stamp-stitch-func


def make_kc_zarr_df(date, run):
    fs_s3 = fsspec.filesystem("s3", anon=False)
    # combined = fs_s3.glob(f"s3://arco-ibf/fcst/gefs_ens/{date}/{run}/gep*")
    year = date[:4]
    month = date[4:6]
    combined = fs_s3.glob(
        f"s3://arco-ibf/fcst/gefs_ens/{year}/{month}/{date}/{run}/gep*"
    )
    combined1 = ["s3://" + f for f in combined]
    mzz = MultiZarrToZarr(
        combined1,
        remote_protocol="s3",
        remote_options={"anon": False},
        concat_dims=["number"],
        identical_dims=["valid_time", "longitude", "latitude"],
    )
    out = mzz.translate()
    fs_ = fsspec.filesystem(
        "reference", fo=out, remote_protocol="s3", remote_options={"anon": True}
    )
    m = fs_.get_mapper("")
    ds = xr.open_dataset(m, engine="zarr", backend_kwargs=dict(consolidated=False))
    return ds


def make_plot_stitch_list(ds, run):
    cont_img_output = []
    cont_full_download_url = []
    cont_animate_flname = []
    avlbl_members = len(ds["number"].values)
    for time in ds["valid_time"].values:
        inner_cont = []
        for member in np.arange(0, avlbl_members):
            date_fmt = pd.to_datetime(str(time))
            file_date_fmt1 = date_fmt.strftime("%Y%m%d%H")
            img_output = f"{file_date_fmt1}_m{member}.jpg"
            inner_cont.append(img_output)
            cont_full_download_url.append(img_output)
            cont_animate_flname.append(f"{file_date_fmt1}.jpg")
        cont_img_output.append(inner_cont)
    date_fmt = pd.to_datetime(str(ds["valid_time"].values[0]))
    year, month = date_fmt.year, date_fmt.month
    month_fmt = str(month).zfill(2)
    date_fmt1 = date_fmt.strftime("%Y%m%d")
    s3_location = f"fcst/gefs_ens/{year}/{month_fmt}/{date_fmt1}/{run}/plot_individual/"
    s3_full_path_urls = [s3_location + f for f in cont_full_download_url]
    animate_flnames = list(set(cont_animate_flname))
    return cont_img_output, s3_full_path_urls, animate_flnames


# @dask.delayed
def s3_download_jpg_file(single_hour_flname, date, run):
    folder_year = date[:4]
    folder_month = date[4:6]
    folder_date = date
    s3_location = f"fcst/gefs_ens/{folder_year}/{folder_month}/{folder_date}/{run}/plot_individual/{single_hour_flname}"
    s3_client = boto3.client("s3")
    local_location = s3_location.split("/")[-1]
    # s3_client.download_file("arco-ibf", s3_location, local_location)
    try:
        s3_client.download_file("arco-ibf", s3_location, local_location)
    except (BotoCoreError, ClientError) as e:
        # Log the error here or pass
        print(f"An error occurred: {e}")
        pass


def clean_files():
    exts = [".jpg", ".mp4"]
    for file in os.listdir("."):
        if not any(file.endswith(ex) for ex in exts):
            os.remove(file)


def make_plot_titles_colorbar(cont_img_output_single_hour, date, run):
    fig = plt.figure(figsize=(11.69, 8.27))  # Adjust the figure size as per your needs
    title_axes = fig.add_axes([0, 0.97, 1, 0.03])
    title_axes.set_axis_off()
    fctimes_axes = fig.add_axes([0, 0.95, 1, 0.03])
    udtimes_axes = fig.add_axes([0, 0.93, 1, 0.03])
    vtimes_axes = fig.add_axes([0, 0.91, 1, 0.03])
    colorbar_axes = fig.add_axes([0.43, 0.95, 0.5, 0.03])
    title_axes.text(
        0.1,
        0.5,
        "Precipitation rate (mm/3hr)",
        fontweight="bold",
        fontsize=10,
        ha="center",
        va="center",
    )
    # Add an axes at the very top of the figure for the start and valid times
    fl_n = cont_img_output_single_hour[0].split("_")[0]
    utc_datetime = datetime.strptime(fl_n, "%Y%m%d%H")

    # Add 3 hours to convert to EAT
    eat_datetime = utc_datetime + timedelta(hours=3)
    str_year = eat_datetime.strftime("%Y")
    str_no_month = eat_datetime.strftime("%m")
    str_day_of_week = eat_datetime.strftime("%a")
    str_month = eat_datetime.strftime("%B")
    str_day = eat_datetime.strftime("%d")
    str_hour = eat_datetime.strftime("%H")
    step = abs(int(utc_datetime.strftime("%H"))) - int(run)
    current_datetime = datetime.now()
    utime_str = current_datetime.strftime("%Y%m%dT%H:%M")
    # Format the EAT datetime as a string

    fctimes_axes.set_axis_off()  # Hide the axes
    fctimes_axes.text(
        0.01,
        0.5,
        f"START TIME: {date} {run} UTC, {str_year}{str_no_month}{str_day} {str_hour} EAT",
        fontsize=10,
        ha="left",
        va="center",
    )
    udtimes_axes.set_axis_off()  # Hide the axes
    udtimes_axes.text(
        0.01,
        0.5,
        f"UPDATE TIME: {utime_str} EAT",
        fontsize=10,
        ha="left",
        va="center",
    )
    vtimes_axes.set_axis_off()  # Hide the axes
    vtimes_axes.text(
        0.01,
        0.5,
        f"VALID TIME: {str_day_of_week} {str_day} {str_month} {str_year} {str_hour} EAT, STEP: {step}",
        fontsize=10,
        ha="left",
        va="center",
    )
    # Define the color bar's colormap
    boundaries = [
        0,
        0.08,
        0.15,
        0.3,
        0.65,
        1.3,
        2.7,
        5.6,
        11.5,
        23.7,
        48.6,
        100,
        205,
        400,
        864,
        1775,
    ]
    colors = [
        "#FFFFFF",  # Adding white color for values below 0.08
        "#00FFFF",
        "#0099FF",
        "#3333FF",
        "#00FF00",
        "#009900",
        "#FFCC00",
        "#FF6600",
        "#FF0000",
        "#CC0000",
        "#990000",
        "#660000",
        "#330000",
        "#FF00FF",
        "#993399",
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", colors, N=len(boundaries) - 1
    )
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_axes,
        orientation="horizontal",
        ticks=boundaries,
    )
    cbar.ax.tick_params(axis="x", labelsize=8)
    labels = [str(boundary) for boundary in boundaries]
    cbar.set_ticklabels(labels)
    return fig, fl_n


@coiled.function(
    memory="16 GiB",
    software="v3-gefs-run-arm-20231113",
    name="funct-plot-stitch",
    region="us-east-1",  # Specific region
    arm=True,  # Change architecture
    idle_timeout="25 minutes",
)
def ens_map_stitich(cont_img_output_single_hour, date, run):
    for single_hour_flname in cont_img_output_single_hour:
        s3_download_jpg_file(single_hour_flname, date, run)
    fig, fl_n = make_plot_titles_colorbar(cont_img_output_single_hour, date, run)
    # grid = ImageGrid(fig, 111, nrows_ncols=(5, 6), axes_pad=0.1)
    grid = ImageGrid(fig, rect=(0, 0, 1, 0.91), nrows_ncols=(4, 8), axes_pad=0.01)
    dpi_value = 300
    width = 346
    height = 486
    # the width and size are determined by a function to check the
    # space available space for stamp plots given the nrows_ncols.
    for i, axGrid in enumerate(grid):
        try:
            image_file = cont_img_output_single_hour[i]
            local_location = f"{image_file}"
        except IndexError:
            local_location = "image_file"
        try:
            image = plt.imread(local_location)
            is_completely_black = (image == 255).all()
            if is_completely_black:
                print(i)
                image = np.zeros((height, width, 3))
                image[image == 0] = 255
        except FileNotFoundError:
            image = np.zeros((height, width, 3))
            image[image == 0] = 255
        original_size = image.shape[:2]  # (height, width)
        target_size = (height, width)  # (height, width)
        # Calculate zoom factors
        zoom_factors = np.array(target_size) / np.array(original_size)
        # Apply zoom (only to the first two dimensions, keeping color channels unchanged if any)
        resized_image = zoom(image, (zoom_factors[0], zoom_factors[1], 1))
        grid[i].imshow(resized_image)
        grid[i].axis("off")
    plt.savefig(f"{fl_n}.jpg", bbox_inches="tight", dpi=150)
    folder_year = date[:4]
    folder_month = date[4:6]
    folder_date = date
    s3_client = boto3.client("s3")
    s3_location = f"fcst/gefs_ens/{folder_year}/{folder_month}/{folder_date}/{run}/plot_stitch/{fl_n}.jpg"
    # s3_location=f"fcst/gefs_ens/{date}/00/plot_individual/{img_output}"
    s3_client.upload_file(f"{fl_n}.jpg", "arco-ibf", s3_location)
    os.remove(f"{fl_n}.jpg")
    # clean_files()
    for single_hour_flname in cont_img_output_single_hour:
        os.remove(single_hour_flname)
    return "stitched and removed in files in memory"


def func_execute_plotting_and_stitching(date, run):
    ds = make_kc_zarr_df(date, run)
    cont_img_output, s3_full_path_urls, animate_flnames = make_plot_stitch_list(ds, run)
    print(cont_img_output)
    ens_map_stitich.cluster.adapt(minimum=1, maximum=2)
    func_env = ens_map_stitich.cluster.get_client()
    func_env.upload_file("utils.py")
    futures = []
    for filename in cont_img_output[0:15]:
        future = ens_map_stitich.submit(filename, date, run)
        futures.append(future)

    results = [f.result() for f in futures]
    return results
