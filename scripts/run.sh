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
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName mpi3deasy  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName mpi3dhard  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName norb  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName causal3d  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName celebahair  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName celebaprimary  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName celebarand  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --seed $seedval
# done

 
############# Dino #############
# python main.py --dsName shapes3d  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName mpi3deasy  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName mpi3dhard  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName norb  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName causal3d  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName celebahair  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName celebaprimary  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName celebarand  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval


############ DRESS with FDAE ############
# python main.py --dsName shapes3d  \
#                --encoder fdae  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName mpi3deasy  \
#                --encoder fdae  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName mpi3dhard  \
#                --encoder fdae  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5   \
#                --seed $seedval

# python main.py --dsName norb  \
#                --encoder fdae  \
#                --imgSizeToEncoder 96 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5   \
#                --seed $seedval

# python main.py --dsName causal3d  \
#                --encoder fdae  \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5   \
#                --seed $seedval

########### DRESS with LSD ############
# python main.py --dsName celebahair  \
#                --encoder lsd  \
#                --imgSizeToEncoder 128 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName celebaprimary  \
#                --encoder lsd  \
#                --imgSizeToEncoder 128 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName celebarand  \
#                --encoder lsd  \
#                --imgSizeToEncoder 128 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval


########### DeepCluster ###########
# python main.py --dsName shapes3d  \
#                --encoder deepcluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval
            
# python main.py --dsName mpi3deasy  \
#                --encoder deepcluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName mpi3dhard  \
#                --encoder deepcluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName norb  \
#                --encoder deepcluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName causal3d \
#                --encoder deepcluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName celebahair  \
#                --encoder deepcluster  \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName celebaprimary  \
#                --encoder deepcluster  \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

# python main.py --dsName celebarand  \
#                --encoder deepcluster  \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --seed $seedval

############ PreTrain and FineTune ############
# python main.py --dsName shapes3d  \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64  \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --seed $seedval

# python main.py --dsName mpi3deasy  \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64  \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --seed $seedval

# python main.py --dsName mpi3dhard  \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64  \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --seed $seedval

# python main.py --dsName norb  \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64  \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --seed $seedval

# python main.py --dsName causal3d  \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64  \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --seed $seedval

# python main.py --dsName celebahair \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 128 --imgSizeToMetaModel 128  \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --seed $seedval

# python main.py --dsName celebaprimary  \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 128 --imgSizeToMetaModel 128  \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --seed $seedval

# python main.py --dsName celebarand  \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 128 --imgSizeToMetaModel 128  \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --seed $seedval

python main.py --dsName causal3d  \
               --encoder metagmvae  \
               --channels 3 --imgSizeToEncoder 64 --imgSizeToMetaModel 64 \
               --NWay 2 --KQuery 5 --KShot 5 \
               --seed $seedval
