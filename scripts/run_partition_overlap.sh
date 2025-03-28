seedval=2000


# Assume the project repo is cloned directly under the user home directory
cd ~/DRESS


####### Supervised meta-learning (and scratch) baselines #####
# for supmethod in "sup" "supall" "supora"; do
#     python main.py --dsName shapes3d  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --seed $seedval \
#                 --computePartitionOverlap
# done


# for supmethod in "sup" "supall" "supora"; do
#     python main.py --dsName mpi3dhard  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --seed $seedval  \
#                 --computePartitionOverlap
# done

# for supmethod in "sup" "supall" "supora"; do
#     python main.py --dsName norb  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --seed $seedval  \
#                 --computePartitionOverlap
# done

# for supmethod in "sup" "supall" "supora"; do
#     python main.py --dsName causal3d  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --seed $seedval  \
#                 --computePartitionOverlap
# done

# for supmethod in "sup" "supall" "supora"; do
#     python main.py --dsName celebahair  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --seed $seedval  \
#                 --computePartitionOverlap
# done

 
############# Dino #############
python main.py --dsName shapes3d  \
               --encoder dino \
               --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --seed $seedval \
               --computePartitionOverlap

python main.py --dsName mpi3dhard  \
               --encoder dino \
               --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --seed $seedval  \
               --computePartitionOverlap

python main.py --dsName norb  \
               --encoder dino \
               --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --seed $seedval  \
               --computePartitionOverlap

python main.py --dsName causal3d  \
               --encoder dino \
               --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --seed $seedval  \
               --computePartitionOverlap

python main.py --dsName celebahair  \
               --encoder dino \
               --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --seed $seedval  \
               --computePartitionOverlap


############ DRESS with FDAE ############
python main.py --dsName shapes3d  \
               --encoder fdae  \
               --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --seed $seedval  \
               --computePartitionOverlap

python main.py --dsName mpi3dhard  \
               --encoder fdae  \
               --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --seed $seedval  \
               --computePartitionOverlap

python main.py --dsName norb  \
               --encoder fdae  \
               --imgSizeToEncoder 96 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5   \
               --seed $seedval  \
               --computePartitionOverlap

python main.py --dsName causal3d  \
               --encoder fdae  \
               --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5   \
               --seed $seedval  \
               --computePartitionOverlap

########### DRESS with LSD ############
python main.py --dsName celebahair  \
               --encoder lsd  \
               --imgSizeToEncoder 128 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --seed $seedval  \
               --computePartitionOverlap


########### DeepCluster ###########
# python main.py --dsName shapes3d  \
#                --encoder deepcluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval  \
#                --computePartitionOverlap

# python main.py --dsName mpi3dhard  \
#                --encoder deepcluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval  \
#                --computePartitionOverlap

# python main.py --dsName norb  \
#                --encoder deepcluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval  \
#                --computePartitionOverlap

# python main.py --dsName causal3d \
#                --encoder deepcluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval  \
#                --computePartitionOverlap

# python main.py --dsName celebahair  \
#                --encoder deepcluster  \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval  \
#                --computePartitionOverlap