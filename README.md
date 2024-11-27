# Kerchunk scan_grib, index file fast refrencing and Streaming of Weather data 

Based on the (dynamic-Grib-chunking method)[https://github.com/asascience-open/nextgen-dmac/commit/6b3286627070c36127ec97b7dbbb88b0ab481f06], the innovative use of Kerchunk scan_grib with
Grib index files significantly reduces the need to scan all Grib files. This
method offers major advantages in reducing the costs involved in
scan_grib—whereas reading all the Grib files in the FMRC to make references
(for example, comparing it with GFS per run, there are 240 hours, or GEFS, which involves 2400 Grib files for 30
members, or ECMWF with 86 files of 4GB files for 50 members) typically requires scanning every file, this approach needs only two file to scans to generate sample metadata instead of scanning of full list of files.

This approach facilitates the creation of references that can be converted into a virtual Zarr dataset.
Utilizing a Dask cluster, this virtual Zarr dataset can be streamed, supporting
transmission and real-time processing through scalable parallel processing.
This capability enables users to access and interact with the data, select
variables, and subset regions and timesteps, which is not feasible with the
Grib data format alone. Although the binary subset method used with an Ensemble
Prediction System dataset is helpful, it does not offer the same level of
flexibility.

Although Grib supports binary subsetting, it remains a method of data
downloading that involves transferring an entire file from a server to a local
device before it can be accessed or used. This method can be compared to the
downloading of MP4 files versus the streaming of video data in HTML5, where a close
comparison can be drawn.

### **Weather Data vs. Video Streaming**
| **Aspect**               | **Video Streaming (HTML5)**                          | **Weather Data (Kerchunk)**                          |
|---------------------------|-----------------------------------------------------|-----------------------------------------------------|
| **Download Workflow**  | Full video download for playback.                   | Full GRIB file download for analysis.               |
| **Streaming Workflow**       | Stream segments on demand using adaptive bitrate.   | Stream slices on demand using Kerchunk metadata.    |
| **Metadata Handling**     | Indexed file for frames, timecodes, and bitrates.   | Indexed metadata for variables, timestamps, region(lat/lon) and Ensemble member etc.    |
| **Efficiency**            | Lower bandwidth; no full downloads needed.          | Lower bandwidth and storage usage.                  |
| **Scalability**           | Scales easily across devices and networks.          | Scales horizontal using Dask cluster DAG|

## GFS in AWS 

Documentation modified [from](https://github.com/fsspec/kerchunk/blob/main/docs/source/reference_aggregation.rst)

Step1: Make virtual dataset for a day in paraquet format

    1. Use kerchunk scan_grib to crete metadata of GFS grib files
    2. Use the metadata mapping to build an index table of every grib message from the .idx files
    3. Combine the index data with the metadata to build any FMRC slice (Horizon, RunTime, ValidTime, BestAvailable)

Step 2: Read paraquet file and stream into zarr, using file 

    1. Paraquet file with 15 variables refences into zarr and store it in GCS 





