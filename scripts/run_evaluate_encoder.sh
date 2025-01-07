# change here for different experimental trials
seedval=2367

# Assume the project repo is cloned directly under the user home directory
cd ~/DRESS

python main.py  --dsName norb  \
                --encoder fdae  \
                --imgSizeToEncoder 96 --imgSizeToMetaModel 84 \
                --NWay 2 --KShot 5 --KQuery 5  \
                --seed $seedval \
                --visualizeTasks