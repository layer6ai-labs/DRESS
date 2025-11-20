seedval=1000


# Assume the project repo is cloned directly under the user home directory
cd ~/DRESS


####### Supervised-All meta-learning baseline #####

# python main.py --dsName norb  \
#                --encoder supall \
#                --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest 5 --KQueryTest 5 \
#                --seed $seedval  \
#                --computePartitionOverlapToMetatest

# python main.py --dsName shapes3d  \
#                --encoder supall \
#                --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest 5 --KQueryTest 5 \
#                --seed $seedval \
#                --computePartitionOverlapToMetatest

# python main.py --dsName causal3d  \
#                --encoder supall \
#                --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest 5 --KQueryTest 5 \
#                --seed $seedval  \
#                --computePartitionOverlapToMetatest

python main.py --dsName mpi3dhard  \
               --encoder supall \
               --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5 \
               --KShotTest 5 --KQueryTest 5 \
               --seed $seedval  \
               --computePartitionOverlapToMetatest

# python main.py --dsName celebahair  \
#                --encoder supall \
#                --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest 5 --KQueryTest 5 \
#                --seed $seedval  \
#                --computePartitionOverlapToMetatest


 
############# Dino #############
# python main.py --dsName shapes3d  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest 5 --KQueryTest 5 \
#                --seed $seedval \
#                --computePartitionOverlapToMetatest

python main.py --dsName mpi3dhard  \
               --encoder dino \
               --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --KShotTest 5 --KQueryTest 5 \
               --seed $seedval  \
               --computePartitionOverlapToMetatest

# python main.py --dsName norb  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest 5 --KQueryTest 5 \
#                --seed $seedval  \
#                --computePartitionOverlapToMetatest

# python main.py --dsName causal3d  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest 5 --KQueryTest 5 \
#                --seed $seedval  \
#                --computePartitionOverlapToMetatest

# python main.py --dsName celebahair  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest 5 --KQueryTest 5 \
#                --seed $seedval  \
#                --computePartitionOverlapToMetatest


############ DRESS with FDAE ############
# python main.py --dsName shapes3d  \
#                --encoder fdae  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest 5 --KQueryTest 5 \
#                --seed $seedval  \
#                --computePartitionOverlapToMetatest

python main.py --dsName mpi3dhard  \
               --encoder fdae  \
               --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --KShotTest 5 --KQueryTest 5 \
               --seed $seedval  \
               --computePartitionOverlapToMetatest

# python main.py --dsName norb  \
#                --encoder fdae  \
#                --imgSizeToEncoder 96 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5   \
#                --KShotTest 5 --KQueryTest 5 \
#                --seed $seedval  \
#                --computePartitionOverlapToMetatest

# python main.py --dsName causal3d  \
#                --encoder fdae  \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5   \
#                --KShotTest 5 --KQueryTest 5 \
#                --seed $seedval  \
#                --computePartitionOverlapToMetatest

########### DRESS with LSD ############
# python main.py --dsName celebahair  \
#                 --encoder lsd  \
#                 --imgSizeToEncoder 128 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5  \
#                 --KShotTest 5 --KQueryTest 5 \
#                 --seed $seedval  \
#                 --computePartitionOverlapToMetatest


########### DeepCluster ###########
# python main.py --dsName norb  \
#                --encoder deepcluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest 5 --KQueryTest 5 \
#                --seed $seedval  \
#                --computePartitionOverlapToMetatest

# python main.py --dsName shapes3d  \
#                --encoder deepcluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest 5 --KQueryTest 5 \
#                --seed $seedval  \
#                --computePartitionOverlapToMetatest

python main.py --dsName mpi3dhard  \
               --encoder deepcluster \
               --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --KShotTest 5 --KQueryTest 5 \
               --seed $seedval  \
               --computePartitionOverlapToMetatest

# python main.py --dsName causal3d \
#                --encoder deepcluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest 5 --KQueryTest 5 \
#                --seed $seedval  \
#                --computePartitionOverlapToMetatest

# python main.py --dsName celebahair  \
#                --encoder deepcluster  \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest 5 --KQueryTest 5 \
#                --seed $seedval  \
#                --computePartitionOverlapToMetatest