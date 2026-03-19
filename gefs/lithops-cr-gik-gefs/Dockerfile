# Lithops Runtime Dockerfile for GEFS Parquet Processing
# =====================================================
#
# This is a Lithops runtime image. Lithops deploys it as a Cloud Run
# service that receives serialized Python functions via GCS and executes them.
#
# IMPORTANT: Unlike the ECMWF Lithops script (which is self-contained),
# the GEFS pipeline requires gefs_util.py baked into the image because
# cloudpickle cannot capture local .py imports on remote workers.
#
# Build with Cloud Build (not local Docker):
#   gcloud builds submit \
#     --config=cloudbuild.yaml \
#     --project=e4drr-crafd \
#     --service-account=projects/e4drr-crafd/serviceAccounts/ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        zip \
    && rm -rf /var/lib/apt/lists/*

# Install lithops + proxy dependencies + GEFS processing dependencies
RUN pip install --no-cache-dir \
        lithops \
        gunicorn \
        flask \
        gevent \
        cloudpickle \
        pika \
        redis \
        requests \
        PyYAML \
        kubernetes \
        numpy \
        ps-mem \
        tblib \
        namegenerator \
        cryptography \
        httplib2 \
        google-cloud-storage \
        google-cloud-pubsub \
        google-api-python-client \
        google-auth \
        psutil \
        pandas \
        pyarrow \
        fsspec \
        s3fs \
        gcsfs \
        kerchunk \
        xarray \
        cfgrib \
        eccodes \
        h5py \
        dask \
        distributed \
        zarr

ENV PORT=8080
ENV PYTHONUNBUFFERED=TRUE
ENV CONCURRENCY=1
ENV TIMEOUT=3600
ENV AWS_NO_SIGN_REQUEST=YES

ENV APP_HOME=/lithops
WORKDIR $APP_HOME

# Copy lithops Cloud Run entry point from installed package
RUN LITHOPS_DIR=$(python -c "import lithops; import os; print(os.path.dirname(lithops.__file__))") && \
    cp ${LITHOPS_DIR}/serverless/backends/gcp_cloudrun/entry_point.py /lithops/lithopsproxy.py

# Bake gefs_util.py into the image so cloudpickle can resolve imports on workers
COPY gefs_util.py /lithops/gefs_util.py
ENV PYTHONPATH=/lithops:$PYTHONPATH

# Pre-download GEFS templates to avoid runtime downloads
# Template tar.gz (~3 MB) contains per-member mapping parquets
# Deflated store parquet (~50 KB) contains zarr metadata skeleton
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    mkdir -p /opt/gefs_templates && \
    curl -L -o /opt/gefs_templates/gik-fmrc-gefs-20241112.tar.gz \
        "https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/resolve/main/gik-fmrc-gefs-20241112.tar.gz" && \
    curl -L -o /opt/gefs_templates/gefs-deflated-store-template-20241112.parquet \
        "https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/resolve/main/gefs-deflated-store-template-20241112.parquet" && \
    echo "Templates downloaded:" && \
    du -h /opt/gefs_templates/* && \
    apt-get remove -y curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

ENV GEFS_TEMPLATE_TAR_GZ=/opt/gefs_templates/gik-fmrc-gefs-20241112.tar.gz
ENV GEFS_TEMPLATE_PARQUET=/opt/gefs_templates/gefs-deflated-store-template-20241112.parquet

CMD exec gunicorn --bind :$PORT --workers $CONCURRENCY --timeout $TIMEOUT lithopsproxy:proxy
