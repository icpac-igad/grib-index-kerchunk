import argparse
import coiled
import dask

from utils import gefs_s3_utl_maker, gen_json
from utils import xcluster_process_kc_individual, func_combine_kc
from utils import acluster_RetryPlotter
from utils import func_execute_plotting_and_stitching


def main_processing(date, run):
    results = xcluster_process_kc_individual(date, run)
    results = func_combine_kc(date, run)

    #plotter = acluster_RetryPlotter(date, run)
    #plotter.run_plotter()
    #plotter.retry(attempt=1)
    #plotter.retry(attempt=2)
    #plotter.shutdown()

    #results = func_execute_plotting_and_stitching(date, run)
    # results = "test"
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process date and run.")
    parser.add_argument("date", type=str, help="Date in the format YYYYMMDD.")
    parser.add_argument("run", type=str, help="Run as a string.")

    args = parser.parse_args()
    main_processing(args.date, args.run)
