# Use the micromamba image as base
FROM docker.io/mambaorg/micromamba:bookworm-slim

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV ENV_NAME="gik"
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"
ARG MAMBA_USER=mambauser

# Set cache and config directories to writable locations
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV XDG_CACHE_HOME=/tmp/.cache
ENV HOME=/tmp

# Install system packages including basic shell and fontconfig
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    curl \
    vim \
    bash \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip
# Create necessary directories with appropriate permissions
RUN mkdir -p $NUMBA_CACHE_DIR $MPLCONFIGDIR $XDG_CACHE_HOME $HOME && \
    chmod -R 777 /tmp

# Set the working directory
#Create directory for AWS credentials
RUN mkdir -p /home/$MAMBA_USER/.aws

# Copy AWS credentials file
COPY aws_credentials /home/$MAMBA_USER/.aws/credentials

# Set proper permissions for the credentials file
#USER root
RUN chown -R $MAMBA_USER:$MAMBA_USER /home/$MAMBA_USER/.aws && \
    chmod 600 /home/$MAMBA_USER/.aws/credentials
USER $MAMBA_USER

# Copy the environment.yml file into the container
COPY devops/environment.yml ./

# Create the environment specified in environment.yml
RUN micromamba create -n $ENV_NAME -f environment.yml \
    && micromamba clean --all --yes

# Set the working directory for the application
WORKDIR /app

# Copy application files to the working directory
RUN git clone https://github.com/nishadhka/grib-index-kerchunk.git /app
#COPY aws_gfs_built_map ./aws_gfs_built_map

# Set the PATH to include the new environment
ENV PATH /opt/conda/envs/$ENV_NAME/bin:$PATH
# Set the default command to run JupyterHub single-user server
#CMD ["jupyterhub-singleuser"]
