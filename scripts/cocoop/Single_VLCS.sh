#!/bin/bash

DATA="/mnt/disk1/theanh28/LAMM/DATA1/VLCS_DATASET"
TRAINER=CoCoOp
#CFG=vit_b16_ep50_ctxv1
DATASET=$1
CFG=$2  # config file

# bash scripts/coop/single_VLCS.sh VLCS
# bash scripts/cocoop/Single_VLCS.sh VLCS vit_b16_c4_ep10_batch1_ctxv1

if [ "$DATASET" = "PACS" ]; then
  ALL_DOMAIN=('art_painting' 'cartoon' 'photo' 'sketch')
elif [ "$DATASET" = "VLCS" ]; then
  ALL_DOMAIN=('caltech' 'labelme' 'pascal' 'sun')
elif [ "$DATASET" = "OfficeHomeDG" ]; then
  ALL_DOMAIN=('art' 'clipart' 'product' 'real_world')
fi

for SHOTS in  8 16  #1 16 
do
    for DOMAIN in "${ALL_DOMAIN[@]}"
    do

        # Tạo một mảng rỗng để lưu các domain đích
        TARGET_DOMAINS_ARRAY=()
        for d in "${ALL_DOMAIN[@]}"
        do
            # Nếu domain đang xét không phải là domain nguồn, thêm vào mảng
            if [ "$d" != "$DOMAIN" ]; then
                TARGET_DOMAINS_ARRAY+=("$d")
            fi
        done
        # Nối các phần tử của mảng thành một chuỗi, phân cách bằng dấu phẩy
        IFS=, TARGET_DOMAINS_STR="${TARGET_DOMAINS_ARRAY[*]}"

        for SEED in  1 
        do
            DIR=output/base/${DATASET}/${TRAINER}/shots_${SHOTS}/${CFG}/seed${SEED}/Single_domain/${DOMAIN}
            if [ -d "$DIR" ]; then
                echo "Results are available in ${DIR}. Resuming..."
                CUDA_VISIBLE_DEVICES=3 python3 -m train \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --source-domains ${DOMAIN} \
                --target-domains ${TARGET_DOMAINS_STR} \
                --shots ${SHOTS} 
            else
                echo "Run this job and save the output to ${DIR}"
                CUDA_VISIBLE_DEVICES=3 python3 -m train \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --source-domains ${DOMAIN} \
                --target-domains ${TARGET_DOMAINS_STR} \
                --shots ${SHOTS} 
            fi
        done   
    done
done
