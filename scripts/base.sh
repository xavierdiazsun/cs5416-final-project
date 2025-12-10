#!/bin/bash

export TOTAL_NODES=3
export NODE_0_IP=132.236.91.179:8101
export NODE_1_IP=132.236.91.179:8102
export NODE_2_IP=132.236.91.179:8103
export FAISS_INDEX_PATH=faiss_index.bin
export DOCUMENTS_DIR=documents/
export USE_GPU=1

./run.sh