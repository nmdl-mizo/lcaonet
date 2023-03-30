#!/bin/bash

# Build the conda package
conda build . -c pytorch -c conda-forge -c nvidia -c pyg

# Install the conda package
conda install --use-local lcaonet
