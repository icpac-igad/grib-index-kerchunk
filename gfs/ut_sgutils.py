import pytest
import os
import tempfile
import pandas as pd
import numpy as np
import xarray as xr

from unittest.mock import patch, MagicMock
#from moto import mock_s3


from sgutils import flatten_list, get_details, foldercreator, gen_json, gefs_s3_utl_maker

from sgutils import (
    xcluster_process_kc_individual,
    gefs_s3_utl_maker,
    gefs_mem_list,
    combine
)
from sgutils import func_combine_kc, acluster_RetryPlotter, make_plot_stitch_list, make_kc_zarr_df
from sgutils import (
    make_plot_stitch_list,
    s3_download_jpg_file,
    clean_files,
    make_plot_titles_colorbar,
    ens_map_stitich,
    func_execute_plotting_and_stitching
)

def test_flatten_list():
    nested_list = [[1, 2], [3, 4], [5, 6]]
    assert flatten_list(nested_list) == [1, 2, 3, 4, 5, 6]
    
    empty_list = []
    assert flatten_list(empty_list) == []
    
    single_level_list = [[1, 2, 3]]
    assert flatten_list(single_level_list) == [1, 2, 3]

def test_get_details():
    url = "s3://noaa-gefs-pds/gefs.20230101/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f003"
    expected = ('20230101', '00', '01', '00', '003')
    assert get_details(url) == expected
    
    invalid_url = "s3://invalid-url"
    assert get_details(invalid_url) is None

def test_foldercreator():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, 'test_folder')
        foldercreator(test_path)
        assert os.path.exists(test_path)
        assert os.path.isdir(test_path)

@patch('sgutils.fsspec.filesystem')
@patch('sgutils.scan_grib')
def test_gen_json(mock_scan_grib, mock_filesystem):
    mock_fs = MagicMock()
    mock_filesystem.return_value = mock_fs
    mock_scan_grib.return_value = [{'key': 'value'}]
    
    s3_url = "s3://noaa-gefs-pds/gefs.20230101/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f003"
    result = gen_json(s3_url)
    
    assert mock_scan_grib.called
    assert mock_fs.open.called
    assert isinstance(result, str)
    assert result.startswith('s3://arco-ibf/fcst/gefs_ens/')

@patch('sgutils.fsspec.filesystem')
def test_gefs_s3_utl_maker(mock_filesystem):
    mock_fs = MagicMock()
    mock_fs.glob.return_value = [
        'noaa-gefs-pds/gefs.20230101/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f003',
        'noaa-gefs-pds/gefs.20230101/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f006'
    ]
    mock_filesystem.return_value = mock_fs
    
    date = '20230101'
    run = '00'
    result = gefs_s3_utl_maker(date, run)
    
    assert mock_fs.glob.called
    assert isinstance(result, list)
    assert all(url.startswith('s3://') for url in result)
    assert len(result) == 2


@patch('sgutils.coiled.Cluster')
@patch('sgutils.gen_json')
@patch('sgutils.gefs_s3_utl_maker')
def test_xcluster_process_kc_individual(mock_gefs_s3_utl_maker, mock_gen_json, mock_cluster):
    # Mock the cluster and client
    mock_client = MagicMock()
    mock_cluster.return_value.get_client.return_value = mock_client
    
    # Mock the gefs_s3_utl_maker function
    mock_gefs_s3_utl_maker.return_value = ['s3://url1', 's3://url2']
    
    # Mock the gen_json function
    mock_gen_json.return_value = 'output_filename'
    
    result = xcluster_process_kc_individual('20230101', '00')
    
    assert mock_cluster.called
    assert mock_client.upload_file.called
    assert mock_gefs_s3_utl_maker.called
    assert mock_gen_json.called
    assert isinstance(result[0], list)
    assert result[0][0] == 'output_filename'

@patch('sgutils.fsspec.filesystem')
def test_gefs_s3_utl_maker(mock_filesystem):
    mock_fs = MagicMock()
    mock_fs.glob.return_value = [
        'noaa-gefs-pds/gefs.20230101/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f003',
        'noaa-gefs-pds/gefs.20230101/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f006'
    ]
    mock_filesystem.return_value = mock_fs
    
    result = gefs_s3_utl_maker('20230101', '00')
    
    assert mock_fs.glob.called
    assert isinstance(result, list)
    assert all(url.startswith('s3://') for url in result)
    assert len(result) == 2

def test_gefs_mem_list():
    lj_glob = [
        's3://noaa-gefs-pds/gefs.20230101/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f003',
        's3://noaa-gefs-pds/gefs.20230101/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f006',
        's3://noaa-gefs-pds/gefs.20230101/00/atmos/pgrb2sp25/gep02.t00z.pgrb2s.0p25.f003'
    ]
    
    result = gefs_mem_list(lj_glob, '20230101', '00', '01')
    
    assert isinstance(result, list)
    assert all('gep01' in url for url in result)
    assert len(result) == 2

@patch('sgutils.fsspec.filesystem')
@patch('sgutils.MultiZarrToZarr')
@patch('tempfile.TemporaryDirectory')
def test_combine(mock_temp_dir, mock_mzz, mock_filesystem):
    # Mock the temporary directory
    mock_temp_dir.return_value.__enter__.return_value = '/tmp/test'
    
    # Mock the filesystem
    mock_fs = MagicMock()
    mock_filesystem.return_value = mock_fs
    
    # Mock MultiZarrToZarr
    mock_mzz_instance = MagicMock()
    mock_mzz_instance.translate.return_value = {'key': 'value'}
    mock_mzz.return_value = mock_mzz_instance
    
    s3_json_urls = [
        's3://arco-ibf/fcst/gefs_ens/2023/01/20230101/00/individual/gep01.t00z.pgrb2s.0p25.f003.json',
        's3://arco-ibf/fcst/gefs_ens/2023/01/20230101/00/individual/gep01.t00z.pgrb2s.0p25.f006.json'
    ]
    
    result = combine(s3_json_urls, '20230101', '00')
    
    assert mock_temp_dir.called
    assert mock_filesystem.called
    assert mock_mzz.called
    assert mock_fs.open.called
    assert result == 'done for a memeber'  # Note: There's a typo in the original function


@patch('sgutils.fsspec.filesystem')
@patch('sgutils.combine.cluster.get_client')
@patch('sgutils.combine.submit')
def test_func_combine_kc(mock_combine_submit, mock_get_client, mock_filesystem):
    # Mock filesystem
    mock_fs = MagicMock()
    mock_fs.glob.return_value = ['arco-ibf/fcst/gefs_ens/2023/01/20230101/00/individual/gep01.t00z.pgrb2s.0p25.f000.json']
    mock_filesystem.return_value = mock_fs

    # Mock combine.cluster.get_client
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Mock combine.submit
    mock_future = MagicMock()
    mock_future.result.return_value = 'Done'
    mock_combine_submit.return_value = mock_future

    result = func_combine_kc('20230101', '00')

    assert mock_filesystem.called
    assert mock_get_client.called
    assert mock_combine_submit.called
    assert mock_client.upload_file.called
    assert len(result) == 30  # Assuming 30 ensemble members

@patch('sgutils.coiled.Cluster')
@patch('sgutils.make_kc_zarr_df')
@patch('sgutils.plot')
def test_acluster_RetryPlotter(mock_plot, mock_make_kc_zarr_df, mock_Cluster):
    # Mock Cluster
    mock_cluster = MagicMock()
    mock_Cluster.return_value = mock_cluster

    # Mock make_kc_zarr_df
    mock_ds = MagicMock()
    mock_ds.sel.return_value = mock_ds
    mock_make_kc_zarr_df.return_value = mock_ds

    # Create instance and run methods
    plotter = acluster_RetryPlotter('20230101', '00')
    plotter.run_plotter()
    plotter.retry()
    plotter.shutdown()

    assert mock_Cluster.called
    assert mock_make_kc_zarr_df.called
    assert mock_plot.called
    assert mock_cluster.get_client.called
    assert mock_cluster.shutdown.called

@patch('sgutils.make_kc_zarr_df')
def test_make_plot_stitch_list(mock_make_kc_zarr_df):
    # Mock the dataset
    mock_ds = MagicMock()
    mock_ds.__getitem__.return_value.values = [pd.Timestamp('2023-01-01')]
    mock_ds['number'].values = range(30)
    mock_make_kc_zarr_df.return_value = mock_ds

    cont_img_output, s3_full_path_urls, animate_flnames = make_plot_stitch_list(mock_ds, '00')

    assert isinstance(cont_img_output, list)
    assert isinstance(s3_full_path_urls, list)
    assert isinstance(animate_flnames, list)
    assert len(cont_img_output) == 1  # One timestamp
    assert len(cont_img_output[0]) == 30  # 30 ensemble members
    assert all('2023010100' in filename for filename in s3_full_path_urls)

@patch('sgutils.fsspec.filesystem')
@patch('sgutils.MultiZarrToZarr')
@patch('sgutils.xr.open_dataset')
def test_make_kc_zarr_df(mock_open_dataset, mock_MultiZarrToZarr, mock_filesystem):
    # Mock filesystem
    mock_fs = MagicMock()
    mock_fs.glob.return_value = ['arco-ibf/fcst/gefs_ens/2023/01/20230101/00/gep01.combined.json']
    mock_filesystem.return_value = mock_fs

    # Mock MultiZarrToZarr
    mock_mzz = MagicMock()
    mock_mzz.translate.return_value = {'key': 'value'}
    mock_MultiZarrToZarr.return_value = mock_mzz

    # Mock xr.open_dataset
    mock_ds = MagicMock()
    mock_open_dataset.return_value = mock_ds

    result = make_kc_zarr_df('20230101', '00')

    assert mock_filesystem.called
    assert mock_MultiZarrToZarr.called
    assert mock_open_dataset.called
    assert result == mock_ds


def test_make_plot_stitch_list():
    # Create a mock dataset
    mock_ds = MagicMock()
    mock_ds['valid_time'].values = [np.datetime64('2023-01-01T00:00:00'), np.datetime64('2023-01-01T03:00:00')]
    mock_ds['number'].values = range(30)

    cont_img_output, s3_full_path_urls, animate_flnames = make_plot_stitch_list(mock_ds, '00')

    assert len(cont_img_output) == 2  # Two timestamps
    assert len(cont_img_output[0]) == 30  # 30 ensemble members
    assert len(s3_full_path_urls) == 60  # 2 timestamps * 30 members
    assert len(animate_flnames) == 2  # Two unique filenames for animation
    assert all('2023010100' in filename for filename in s3_full_path_urls[:30])
    assert all('2023010103' in filename for filename in s3_full_path_urls[30:])

@patch('sgutils.boto3.client')
def test_s3_download_jpg_file(mock_boto3_client):
    mock_s3_client = MagicMock()
    mock_boto3_client.return_value = mock_s3_client

    s3_download_jpg_file('2023010100_m00.jpg', '20230101', '00')

    mock_s3_client.download_file.assert_called_once()
    args = mock_s3_client.download_file.call_args[0]
    assert args[0] == 'arco-ibf'
    assert '2023010100_m00.jpg' in args[1]
    assert args[2] == '2023010100_m00.jpg'

@patch('sgutils.os.remove')
@patch('sgutils.os.listdir')
def test_clean_files(mock_listdir, mock_remove):
    mock_listdir.return_value = ['file1.txt', 'file2.jpg', 'file3.mp4', 'file4.png']
    
    clean_files()

    assert mock_remove.call_count == 2
    mock_remove.assert_any_call('file1.txt')
    mock_remove.assert_any_call('file4.png')

@patch('sgutils.plt.figure')
@patch('sgutils.plt.savefig')
def test_make_plot_titles_colorbar(mock_savefig, mock_figure):
    mock_fig = MagicMock()
    mock_figure.return_value = mock_fig

    cont_img_output_single_hour = ['2023010100_m00.jpg', '2023010100_m01.jpg']
    fig, fl_n = make_plot_titles_colorbar(cont_img_output_single_hour, '20230101', '00')

    assert fig == mock_fig
    assert fl_n == '2023010100'
    assert mock_savefig.called

@patch('sgutils.s3_download_jpg_file')
@patch('sgutils.make_plot_titles_colorbar')
@patch('sgutils.ImageGrid')
@patch('sgutils.plt.savefig')
@patch('sgutils.boto3.client')
@patch('sgutils.os.remove')
def test_ens_map_stitich(mock_remove, mock_boto3_client, mock_savefig, mock_ImageGrid, mock_make_plot_titles_colorbar, mock_s3_download_jpg_file):
    mock_make_plot_titles_colorbar.return_value = (MagicMock(), '2023010100')
    mock_ImageGrid.return_value = [MagicMock() for _ in range(32)]
    
    cont_img_output_single_hour = [f'2023010100_m{i:02d}.jpg' for i in range(30)]
    result = ens_map_stitich(cont_img_output_single_hour, '20230101', '00')

    assert mock_s3_download_jpg_file.call_count == 30
    assert mock_make_plot_titles_colorbar.called
    assert mock_ImageGrid.called
    assert mock_savefig.called
    assert mock_boto3_client.return_value.upload_file.called
    assert mock_remove.call_count == 31  # 30 input files + 1 output file
    assert result == 'stitched and removed in files in memory'

@patch('sgutils.make_kc_zarr_df')
@patch('sgutils.make_plot_stitch_list')
@patch('sgutils.ens_map_stitich.cluster.adapt')
@patch('sgutils.ens_map_stitich.cluster.get_client')
@patch('sgutils.ens_map_stitich.submit')
def test_func_execute_plotting_and_stitching(mock_submit, mock_get_client, mock_adapt, mock_make_plot_stitch_list, mock_make_kc_zarr_df):
    mock_ds = MagicMock()
    mock_make_kc_zarr_df.return_value = mock_ds

    mock_make_plot_stitch_list.return_value = (
        [['file1.jpg', 'file2.jpg'] for _ in range(15)],
        ['s3://path/to/file1.jpg', 's3://path/to/file2.jpg'],
        ['animate1.jpg', 'animate2.jpg']
    )

    mock_future = MagicMock()
    mock_future.result.return_value = 'Done'
    mock_submit.return_value = mock_future

    result = func_execute_plotting_and_stitching('20230101', '00')

    assert mock_make_kc_zarr_df.called
    assert mock_make_plot_stitch_list.called
    assert mock_adapt.called
    assert mock_get_client.called
    assert mock_submit.call_count == 15
    assert len(result) == 15
    assert all(r == 'Done' for r in result)


if __name__ == '__main__':
    pytest.main()
