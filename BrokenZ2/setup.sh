#!/bin/bash

conda deactivate
conda activate py2
export PYTHONPATH="${PYTHONPATH}:${PWD}/cosmoTransitions"
