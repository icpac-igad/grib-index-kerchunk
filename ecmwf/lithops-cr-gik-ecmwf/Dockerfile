# Lithops Runtime Dockerfile for ECMWF Parquet Processing
# ======================================================
#
# This is a Lithops runtime image. Lithops deploys it as a Cloud Run
# service that receives serialized Python functions via GCS and executes them.
#
# Build with Cloud Build (not local Docker):
#   gcloud builds submit \
#     --config=cloudbuild.yaml \
#     --project=e4drr-crafd \
#     --service-account=projects/e4drr-crafd/serviceAccounts/ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com
#
# The image installs lithops via pip and copies its Cloud Run entry point,
# eliminating the need for 'lithops runtime build' (which requires local Docker).

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        zip \
    && rm -rf /var/lib/apt/lists/*

# Install lithops + proxy dependencies + processing dependencies in one layer
RUN pip install --no-cache-dir \
        lithops==3.6.4 \
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
        gcsfs

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

# Pre-bake the per-era ECMWF template (constant, shared across all workers).
# Era selected at BUILD time via --build-arg; matching runtime env set below
# (overridable at deploy). Defaults = 50r1 (current ongoing ops).
#   49r1 13-level : --build-arg TEMPLATE_ARTIFACT=gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz \
#                   --build-arg ECMWF_REFERENCE_DATE=20250515 --build-arg ECMWF_CONTROL_STREAM=enfo
#   50r1 14-level : gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz (default; 20260513, oper control)
#   0.4-beta 9lvl : --build-arg TEMPLATE_ARTIFACT=gik-fmrc-v2ecmwf_fmrc-0p4-beta.tar.gz \
#                   --build-arg ECMWF_REFERENCE_DATE=20230601 --build-arg ECMWF_RESOLUTION=0p4 \
#                   --build-arg ECMWF_CONTROL_STREAM=enfo
ARG TEMPLATE_REPO=E4DRR/grib-index-kerchunk-templates
ARG TEMPLATE_ARTIFACT=gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz
ARG ECMWF_REFERENCE_DATE=20260513
ARG ECMWF_RESOLUTION=0p25
ARG ECMWF_CONTROL_STREAM=oper
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    mkdir -p /opt/ecmwf_templates && \
    curl -L -o /opt/ecmwf_templates/${TEMPLATE_ARTIFACT} \
        "https://huggingface.co/datasets/${TEMPLATE_REPO}/resolve/main/${TEMPLATE_ARTIFACT}" && \
    echo "Template baked: ${TEMPLATE_ARTIFACT} ($(du -h /opt/ecmwf_templates/${TEMPLATE_ARTIFACT} | cut -f1))" && \
    apt-get remove -y curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Runtime env matching the baked era (overridable at `lithops runtime deploy`).
ENV ECMWF_TEMPLATE_PATH=/opt/ecmwf_templates/${TEMPLATE_ARTIFACT}
ENV TEMPLATE_URL=https://huggingface.co/datasets/${TEMPLATE_REPO}/resolve/main/${TEMPLATE_ARTIFACT}
ENV ECMWF_REFERENCE_DATE=${ECMWF_REFERENCE_DATE}
ENV ECMWF_RESOLUTION=${ECMWF_RESOLUTION}
ENV ECMWF_CONTROL_STREAM=${ECMWF_CONTROL_STREAM}

CMD exec gunicorn --bind :$PORT --workers $CONCURRENCY --timeout $TIMEOUT lithopsproxy:proxy
