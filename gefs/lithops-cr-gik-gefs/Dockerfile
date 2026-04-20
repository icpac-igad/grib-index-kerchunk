# Lithops Runtime Dockerfile for GEFS Parquet Processing
# ======================================================
#
# This is a Lithops runtime image. Lithops deploys it as a Cloud Run
# service that receives serialized Python functions via GCS and executes them.
#
# Build with Cloud Build (not local Docker):
#   gcloud builds submit \
#     --config=cloudbuild.yaml \
#     --project=e4drr-crafd \
#     --service-account=projects/e4drr-crafd/serviceAccounts/gefs-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com
#
# The image installs lithops via pip and copies its Cloud Run entry point,
# eliminating the need for 'lithops runtime build' (which requires local Docker).

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        zip \
    && rm -rf /var/lib/apt/lists/*

# Install lithops + proxy dependencies + processing dependencies + kerchunk for GRIB in one layer
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
        zarr

ENV PORT=8080
ENV PYTHONUNBUFFERED=TRUE
ENV CONCURRENCY=1
ENV TIMEOUT=3600
ENV AWS_NO_SIGN_REQUEST=YES

ENV APP_HOME=/lithops
WORKDIR $APP_HOME

# Copy lithops Cloud Run entry point from installed package
# This replaces the lithops_cloudrun.zip approach
RUN LITHOPS_DIR=$(python -c "import lithops; import os; print(os.path.dirname(lithops.__file__))") && \
    cp ${LITHOPS_DIR}/serverless/backends/gcp_cloudrun/entry_point.py /lithops/lithopsproxy.py

# Pre-download GEFS template to avoid runtime downloads
# This 120MB file is constant and shared across all workers
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    mkdir -p /opt/gefs_templates && \
    curl -L --retry 5 --retry-delay 5 --retry-all-errors \
        -o /opt/gefs_templates/gik-fmrc-gefs-20241112.tar.gz \
        "https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/resolve/main/gik-fmrc-gefs-20241112.tar.gz" && \
    echo "Template downloaded: $(du -h /opt/gefs_templates/gik-fmrc-gefs-20241112.tar.gz | cut -f1)" && \
    apt-get remove -y curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Bake the deflated store template parquet (27KB) directly into the image
COPY gefs-deflated-store-template-20241112.parquet /opt/gefs_templates/

ENV GEFS_TEMPLATE_TAR_GZ=/opt/gefs_templates/gik-fmrc-gefs-20241112.tar.gz
ENV GEFS_TEMPLATE_PARQUET=/opt/gefs_templates/gefs-deflated-store-template-20241112.parquet

CMD exec gunicorn --bind :$PORT --workers $CONCURRENCY --timeout $TIMEOUT lithopsproxy:proxy
