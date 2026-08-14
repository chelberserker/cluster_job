#!/bin/bash

ENV_NAME="SANS_env"

# Create the environment and install packages from conda-forge
mamba create -n $ENV_NAME -c conda-forge -y \
    python=3.10 \
    numpy \
    scipy \
    matplotlib \
    h5py \
    lmfit \
    corner \
    jscatter \
    pip  \
    ScatTools

# Initialize mamba for the current script session to allow activation
eval "$(mamba shell hook --shell bash)"
mamba activate $ENV_NAME

echo "Environment $ENV_NAME successfully created and ready."