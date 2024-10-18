# change here for different experimental trials
seedval=2367

# Assume the project repo is cloned directly under the user home directory
cd ~/Diversified_Tasks_Meta_Learning

python main.py  --dsName mpi3dtoyhard  \
                --encoder fdae  \
                --numEncodingPartitions -1  \
                --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
                --seed $seedval \
                --visualizeTasks