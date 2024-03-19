#!/bin/bash

CUDA=0
CONFIG_PATH="./maskingdepth/conf/base_dust3r_kitti.yaml"

CUDA_VISIBLE_DEVICES=${CUDA} python usage.py --conf ${CONFIG_PATH} 