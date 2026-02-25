# CFS Lithops Cloud Run Runtime
# ==============================
#
# Template-based pipeline — no kerchunk/eccodes needed on workers.
# Parses CFS .idx files directly (colon-separated text).
#
# Build:
#   gcloud builds submit \
#     --config=cloudbuild.yaml \
#     --project=e4drr-crafd \
#     --service-account=projects/e4drr-crafd/serviceAccounts/cfs-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com
#
# Deploy:
#   lithops runtime deploy gcr.io/e4drr-crafd/cfs-lithops-runtime \
#     -b gcp_cloudrun -s gcp_storage --config lithops_config.yaml

FROM python:3.11-slim-bookworm

# Install Python dependencies — no kerchunk/eccodes needed
RUN pip install --no-cache-dir \
    lithops \
    pandas \
    numpy \
    pyarrow \
    fsspec \
    s3fs \
    gcsfs \
    requests \
    google-cloud-storage \
    google-cloud-pubsub \
    google-api-python-client \
    google-auth \
    httplib2

# Bake the CFS template into the image so workers don't need to download it
# Template must be built first via run_cfs_template_creation.py and placed here
COPY gik-fmrc-v2cfs_fmrc.tar.gz /opt/cfs-template/gik-fmrc-v2cfs_fmrc.tar.gz
ENV CFS_TEMPLATE_PATH=/opt/cfs-template/gik-fmrc-v2cfs_fmrc.tar.gz

# Lithops Cloud Run entry point
ENTRYPOINT ["python", "-m", "lithops.serverless.backends.gcp_cloudrun.entry_point"]
