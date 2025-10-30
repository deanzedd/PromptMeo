#!/bin/bash

#cd ../..

# custom config
DATA="/mnt/disk1/theanh28/LAMM/DATA1"
TRAINER=ZeroshotCLIP
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16

# bash scripts/zsclip/zeroshot.sh PACS vit_b16
# bash scripts/zsclip/zeroshot.sh VLCS vit_b16
# bash scripts/zsclip/zeroshot.sh OfficeHomeDG vit_b16

CUDA_VISIBLE_DEVICES=1 python3 -m train \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml \
    --output-dir output/${TRAINER}/${CFG}/${DATASET} \
    --eval-only