#!/bin/bash

DATA="/mnt/disk1/theanh28/LAMM/DATA1/PACS_DATASET"
TRAINER=CoOp
#CFG=vit_b16_ep50_ctxv1
DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
#SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$5  # class-specific context (False or True)
# bash scripts/coop/single_VLCS.sh VLCS
# bash scripts/coop/Single_PACS.sh PACS vit_b16_ep50_ctxv1 end 4 False

if [ "$DATASET" = "PACS" ]; then
  ALL_DOMAIN=('art_painting' 'cartoon' 'photo' 'sketch')
elif [ "$DATASET" = "VLCS" ]; then
  ALL_DOMAIN=('caltech' 'labelme' 'pascal' 'sun')
elif [ "$DATASET" = "OfficeHomeDG" ]; then
  ALL_DOMAIN=('art' 'clipart' 'product' 'real_world')
fi

for SHOTS in 1 4 8 16  #1 16 
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
                CUDA_VISIBLE_DEVICES=2 python3 -m train \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --source-domains ${DOMAIN} \
                --target-domains ${TARGET_DOMAINS_STR} \
                --shots ${SHOTS} \
                #DATASET.NUM_SHOTS ${SHOTS} 
                #--shots ${SHOTS} \
                #--triplet-loss \
                #TRAINER.COOP.N_CTX ${NCTX} \
                
                #TRAINER.COOP.CSC ${CSC} \
                #TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                #DATASET.NUM_SHOTS ${SHOTS}
            else
                echo "Run this job and save the output to ${DIR}"
                CUDA_VISIBLE_DEVICES=2 python3 -m train \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --source-domains ${DOMAIN} \
                --target-domains ${TARGET_DOMAINS_STR} \
                --shots ${SHOTS} \
                #DATASET.NUM_SHOTS ${SHOTS} 
                #--triplet-loss \
            fi
        done   
    done
done
