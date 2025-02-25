#!/bin/bash

datasets=("shapes3d")
random_integers=(5912 8224 4225 3153 8160 906 2355 3000 4600 9430)

for seed in "${random_integers[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "Running training on dataset: $dataset with seed: $seed"
        python train.py \
            --num-epochs 100 \
            --model psco \
            --backbone conv4 \
            --prediction \
            --num-shots 1 \
            --dataset "" \
            --datadir "" \
            --logdir outputs \
            --dsName "$dataset" \
            --encoder "sup" \
            --imgSizeToEncoder "-1" \
            --imgSizeToMetaModel "64" \
            --NWay 2 \
            --KShot 5 \
            --KQuery 5 \
            --device-id 0 \
            --seed "$seed"
    done
done