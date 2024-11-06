# change here for different experimental trials
seedval=2367

# Assume the project repo is cloned directly under the user home directory
cd ~/DRESS

python main.py  --dsName celebahard  \
                --encoder fdae  \
                --numEncodingPartitions -1  \
                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
                --seed $seedval \
                --visualizeTasks