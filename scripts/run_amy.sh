# change here for different experimental trials
seedval=3278


# Assume the project repo is cloned directly under the user home directory
cd ~/DRESS

python main.py --dsName causal3d \
               --encoder deepcluster \
               --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --seed $seedval

python main.py --dsName causal3d \
               --encoder ablate_disentangle \
               --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5 \
               --seed $seedval

python main.py --dsName celebahair \
               --encoder ablate_align \
               --imgSizeToEncoder 128 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5 \
               --seed $seedval