# change here for different experimental trials
seedval=3278


# Assume the project repo is cloned directly under the user home directory
cd ~/DRESS

python main.py --dsName norb  \
               --encoder dino \
               --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --seed $seedval
            
python main.py --dsName causal3d  \
               --encoder dino \
               --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --seed $seedval

python main.py --dsName norb  \
               --encoder deepcluster \
               --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --seed $seedval
