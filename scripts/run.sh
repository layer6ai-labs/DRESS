# change here for different experimental trials
seedval=1234
# seedval=1000
# seedval=2367

# Assume the project repo is cloned directly under the user home directory
cd ~/DRESS

####### Supervised meta-learning (and scratch) baselines #####
# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName shapes3d  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5 \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName mpi3dtoy  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5 \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName mpi3dtoyhard  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5 \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName celebarand  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5 \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName celebahard  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5 \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName animals  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5 \
#                 --seed $seedval
# done
 

######### Dino #########
# python main.py --dsName shapes3d  \
#                --encoder dino \
#                --numEncodingPartitions 50 \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName mpi3dtoy  \
#                --encoder dino \
#                --numEncodingPartitions 50 \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName mpi3dtoyhard  \
#                --encoder dino \
#                --numEncodingPartitions 50 \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName celebarand  \
#                --encoder dino \
#                --numEncodingPartitions 50 \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName celebahair  \
#                --encoder dino \
#                --numEncodingPartitions 50 \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName celebaeyes  \
#                --encoder dino \
#                --numEncodingPartitions 50 \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName animals  \
#                --encoder dino \
#                --numEncodingPartitions 50 \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
#                --seed $seedval

#######DeepCluster#######
# python main.py --dsName shapes3d  \
#                --encoder deepcluster \
#                --numEncodingPartitions 50 \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
#                --seed $seedval
            
# python main.py --dsName mpi3dtoy  \
#                --encoder deepcluster \
#                --numEncodingPartitions 50 \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName mpi3dtoyhard  \
#                --encoder deepcluster \
#                --numEncodingPartitions 50 \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName celebarand  \
#                --encoder deepcluster \
#                --numEncodingPartitions 50 \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName celebahard  \
#                --encoder deepcluster \
#                --numEncodingPartitions 50 \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
#                --seed $seedval

#######DRESS with FDAE########
# python main.py --dsName shapes3d  \
#                --encoder fdae  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName mpi3dtoy  \
#                --encoder fdae  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName mpi3dtoyhard  \
#                --encoder fdae  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5   \
#                --seed $seedval

# python main.py --dsName celebarand  \
#                --encoder fdae  \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5   \
#                --seed $seedval

# python main.py --dsName celebahair  \
#                --encoder fdae  \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5   \
#                --seed $seedval

python main.py --dsName celebaeyes  \
               --encoder fdae  \
               --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
               --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5   \
               --seed $seedval

# python main.py --dsName animals  \
#                --encoder fdae  \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5   \
#                --seed $seedval

#######DRESS with SODA########
# python main.py --dsName shapes3d  \
#                --encoder soda  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName mpi3dtoy  \
#                --encoder soda  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName mpi3dtoyhard  \
#                --encoder soda  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5   \
#                --seed $seedval

# python main.py --dsName celebahard  \
#                --encoder soda  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShotMetaTr 5 --KShotMetaVa 5 --KShotMetaTe 5 --KQuery 5   \
#                --seed $seedval


######PreTrain and FineTune######
# python main.py --dsName shapes3d  \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64  \
#                --NWay 2 --KShotMetaTe 5 --KQuery 5 \
#                --seed $seedval

# python main.py --dsName mpi3dtoy  \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64  \
#                --NWay 2 --KShotMetaTe 5 --KQuery 5 \
#                --seed $seedval

# python main.py --dsName mpi3dtoyhard  \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64  \
#                --NWay 2 --KShotMetaTe 5 --KQuery 5 \
#                --seed $seedval
