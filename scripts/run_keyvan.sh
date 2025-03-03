# change here for different experimental trials
seedval=3278


# Assume the project repo is cloned directly under the user home directory
cd ~/DRESS

for supmethod in "sup" "supall" "supora" "scratch"; do
    python main.py --dsName norb  \
                --encoder $supmethod \
                --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
                --NWay 2 --KShot 5 --KQuery 5 \
                --seed $seedval
done