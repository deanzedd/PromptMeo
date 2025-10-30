#!/bin/bash

DATA="/mnt/disk1/theanh28/LAMM/DATA1"
TRAINER=PromptMeo
#CFG=vit_b16_ep50_ctxv1
DATASET=OfficeHomeDG
CFG=vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets  # config file

# bash scripts/coop/single_VLCS.sh VLCS
# bash scripts/promptsrc/Multi_officeDG.sh 
# bash scripts/promptsrc/Single_PACS.sh PACS vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets

if [ "$DATASET" = "PACS" ]; then
  ALL_DOMAIN=('art_painting' 'cartoon' 'photo' 'sketch')
elif [ "$DATASET" = "VLCS" ]; then
  ALL_DOMAIN=('caltech' 'labelme' 'pascal' 'sun')
elif [ "$DATASET" = "OfficeHomeDG" ]; then
  ALL_DOMAIN=('art' 'clipart' 'product' 'real_world')
fi

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
        DIR=output/base/${DATASET}/${TRAINER}/${CFG}/seed${SEED}/Multi_domain/${DOMAIN}
        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Resuming..."
            CUDA_VISIBLE_DEVICES=3 python3 -m train \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --source-domains ${TARGET_DOMAINS_STR} \
            --target-domains ${DOMAIN}
        else
            echo "Run this job and save the output to ${DIR}"
            CUDA_VISIBLE_DEVICES=3 python3 -m train \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --source-domains ${TARGET_DOMAINS_STR} \
            --target-domains ${DOMAIN}
        fi
    done   
done

