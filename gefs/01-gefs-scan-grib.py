import argparse
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

from utils import gefs_s3_utl_maker, gen_json
from utils import xcluster_process_kc_individual, func_combine_kc
from utils import acluster_RetryPlotter
from utils import func_execute_plotting_and_stitching

@task
def process_kc_individual(date: str, run: str):
    return xcluster_process_kc_individual(date, run)

@task
def combine_kc(date: str, run: str):
    return func_combine_kc(date, run)

@task
def run_plotter(date: str, run: str):
    plotter = acluster_RetryPlotter(date, run)
    plotter.run_plotter()
    plotter.retry(attempt=1)
    plotter.retry(attempt=2)
    plotter.shutdown()

@task
def execute_plotting_and_stitching(date: str, run: str):
    return func_execute_plotting_and_stitching(date, run)

@flow(task_runner=SequentialTaskRunner())
def main_processing(date: str, run: str):
    process_kc_individual.submit(date, run)
    combine_kc.submit(date, run)
    run_plotter.submit(date, run)
    execute_plotting_and_stitching.submit(date, run)
    return "test"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process date and run.")
    parser.add_argument("date", type=str, help="Date in the format YYYYMMDD.")
    parser.add_argument("run", type=str, help="Run as a string.")

    args = parser.parse_args()
    main_processing(args.date, args.run)
