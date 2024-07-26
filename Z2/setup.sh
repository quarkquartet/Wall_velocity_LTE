#!/bin/bash

conda deactivate
conda activate py3-10-9
export PYTHONPATH="${PYTHONPATH}:${PWD}/model"
