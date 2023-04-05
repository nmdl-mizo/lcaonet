#!/bin/bash

# Build the conda package
conda build . -c pytorch -c conda-forge -c pyg

# Install the conda package
conda install --use-local lcaonet
