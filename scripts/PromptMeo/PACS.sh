#!/bin/bash
#set -x
DATA="/home/aidev/dungnt/thanh/PromptMeo/DATA"
TRAINER=PromptMeo
#CFG=vit_b16_ep50_ctxv1
DATASET=PACS_SF
CFG=20epoch  # config file
#echo $SHELL
#echo $BASH_VERSION
#/mnt/disk1/theanh28/PromptMeo/configs/trainers/PromptMeo/vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets.yaml
# bash scripts/PromptMeo/PACS.sh
# bash scripts/promptsrc/Multi_officeDG.sh 
# bash scripts/promptsrc/Single_PACS.sh PACS vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets
# CUDA_VISIBLE_DEVICES=3

if [ "$DATASET" = "PACS" ]; then
  ALL_DOMAIN=('art_painting' 'cartoon' 'photo' 'sketch')
elif [ "$DATASET" = "PACS_SF" ]; then
  ALL_DOMAIN=('art_painting' 'cartoon' 'photo' 'sketch')
elif [ "$DATASET" = "VLCS" ]; then
  ALL_DOMAIN=('caltech' 'labelme' 'pascal' 'sun')
elif [ "$DATASET" = "OfficeHomeDG" ]; then
  ALL_DOMAIN=('art' 'clipart' 'product' 'real_world')
fi


for SEED in  1 
do
    DIR=output/base/${DATASET}/${TRAINER}/${CFG}/seed${SEED}/Multi_domain/sketch
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Resuming..."
        CUDA_VISIBLE_DEVICES=0 python3 -m train \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --source-domains art_painting cartoon photo \
        --target-domains sketch \
        --num_styles 80 --txts_path dassl/txts
    else
        echo "Run this job and save the output to ${DIR}"
        CUDA_VISIBLE_DEVICES=0 python3 -m train \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --source-domains art_painting cartoon photo \
        --target-domains sketch \
        --num_styles 80 --txts_path dassl/txts
    fi
done   


