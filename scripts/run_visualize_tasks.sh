# change here for different experimental trials
seedval=1234

# Assume the project repo is cloned directly under the user home directory
cd ~/DRESS

python main.py  --dsName celebaprimary  \
                --encoder lsd  \
                --imgSizeToEncoder 128 --imgSizeToMetaModel 84 \
                --NWay 2 --KShot 5 --KQuery 5  \
                --seed $seedval \
                --visualizeTasks