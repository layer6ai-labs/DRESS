# change here for different experimental trials
seedval=3278
kShotTest=1
kQueryTest=5


# Assume the project repo is cloned directly under the user home directory
cd ~/DRESS

####### Supervised meta-learning (and scratch) baselines #####
# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName shapes3d  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName mpi3deasy  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName mpi3dhard  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName norb  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName causal3d  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName celebahair  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName celebaprimary  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                 --seed $seedval
# done

# for supmethod in "sup" "supall" "supora" "scratch"; do
#     python main.py --dsName celebarand  \
#                 --encoder $supmethod \
#                 --imgSizeToEncoder -1 --imgSizeToMetaModel 84 \
#                 --NWay 2 --KShot 5 --KQuery 5 \
#                 --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                 --seed $seedval
# done

 
# ############# Dino #############
# python main.py --dsName shapes3d  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName mpi3deasy  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName mpi3dhard  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName norb  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName causal3d  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName celebahair  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName celebaprimary  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName celebarand  \
#                --encoder dino \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval


############ DRESS with FDAE ############
python main.py --dsName shapes3d  \
               --encoder fdae  \
               --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --KShotTest $kShotTest --KQueryTest $kQueryTest \
               --seed $seedval

python main.py --dsName mpi3deasy  \
               --encoder fdae  \
               --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --KShotTest $kShotTest --KQueryTest $kQueryTest \
               --seed $seedval

python main.py --dsName mpi3dhard  \
               --encoder fdae  \
               --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5   \
               --KShotTest $kShotTest --KQueryTest $kQueryTest \
               --seed $seedval

python main.py --dsName norb  \
               --encoder fdae  \
               --imgSizeToEncoder 96 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5   \
               --KShotTest $kShotTest --KQueryTest $kQueryTest \
               --seed $seedval

python main.py --dsName causal3d  \
               --encoder fdae  \
               --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5   \
               --KShotTest $kShotTest --KQueryTest $kQueryTest \
               --seed $seedval

########## DRESS with LSD ############
python main.py --dsName celebahair  \
               --encoder lsd  \
               --imgSizeToEncoder 128 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --KShotTest $kShotTest --KQueryTest $kQueryTest \
               --seed $seedval

python main.py --dsName celebaprimary  \
               --encoder lsd  \
               --imgSizeToEncoder 128 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --KShotTest $kShotTest --KQueryTest $kQueryTest \
               --seed $seedval

python main.py --dsName celebarand  \
               --encoder lsd  \
               --imgSizeToEncoder 128 --imgSizeToMetaModel 84 \
               --NWay 2 --KShot 5 --KQuery 5  \
               --KShotTest $kShotTest --KQueryTest $kQueryTest \
               --seed $seedval


########### DeepCluster ###########
# python main.py --dsName shapes3d  \
#                --encoder deepcluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval
            
# python main.py --dsName mpi3deasy  \
#                --encoder deepcluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName mpi3dhard  \
#                --encoder deepcluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName norb  \
#                --encoder deepcluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName causal3d \
#                --encoder deepcluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName celebahair  \
#                --encoder deepcluster  \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName celebaprimary  \
#                --encoder deepcluster  \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName celebarand  \
#                --encoder deepcluster  \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5  \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

############ PreTrain and FineTune ############
# python main.py --dsName shapes3d  \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64  \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName mpi3deasy  \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64  \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName mpi3dhard  \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64  \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName norb  \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64  \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName causal3d  \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64  \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName celebahair \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 128 --imgSizeToMetaModel 128  \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName celebaprimary  \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 128 --imgSizeToMetaModel 128  \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName celebarand  \
#                --encoder simclrpretrain  \
#                --imgSizeToEncoder 128 --imgSizeToMetaModel 128  \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

############## Meta-GMVAE #############
# python main.py --dsName shapes3d  \
#                --encoder metagmvae  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64 \
#                --NWay 2 --KQuery 5 --KShot 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName causal3d  \
#                --encoder metagmvae  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64 \
#                --NWay 2 --KQuery 5 --KShot 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval
    
# python main.py --dsName norb  \
#                --encoder metagmvae  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64 \
#                --NWay 2 --KQuery 5 --KShot 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval
            
# python main.py --dsName mpi3deasy  \
#                --encoder metagmvae  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64 \
#                --NWay 2 --KQuery 5 --KShot 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval
    
# python main.py --dsName mpi3dhard  \
#                --encoder metagmvae  \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 64 \
#                --NWay 2 --KQuery 5 --KShot 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName celebahair  \
#                --encoder metagmvae  \
#                --imgSizeToEncoder 128 --imgSizeToMetaModel 128 \
#                --NWay 2 --KQuery 5 --KShot 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName celebaprimary  \
#                --encoder metagmvae  \
#                --imgSizeToEncoder 128 --imgSizeToMetaModel 128 \
#                --NWay 2 --KQuery 5 --KShot 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval
            
# python main.py --dsName celebarand  \
#                --encoder metagmvae  \
#                --imgSizeToEncoder 128 --imgSizeToMetaModel 128 \
#                --NWay 2 --KQuery 5 --KShot 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

############### Ablation I ##############
# python main.py --dsName shapes3d \
#                --encoder ablate_disentangle \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName causal3d \
#                --encoder ablate_disentangle \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName mpi3dhard \
#                --encoder ablate_disentangle \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName celebahair \
#                --encoder ablate_disentangle \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName celebaprimary \
#                --encoder ablate_disentangle \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

############### Ablation II ##############
# python main.py --dsName celebahair \
#                --encoder ablate_align \
#                --imgSizeToEncoder 128 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName celebaprimary \
#                --encoder ablate_align \
#                --imgSizeToEncoder 128 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

############### Ablation III ##############
# python main.py --dsName shapes3d \
#                --encoder ablate_individual_cluster \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName causal3d \
#                --encoder ablate_individual_cluster \
#                --imgSizeToEncoder 224 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName mpi3dhard \
#                --encoder ablate_individual_cluster \
#                --imgSizeToEncoder 64 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval

# python main.py --dsName celebahair \
#                --encoder ablate_individual_cluster \
#                --imgSizeToEncoder 128 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval
            
# python main.py --dsName celebaprimary \
#                --encoder ablate_individual_cluster \
#                --imgSizeToEncoder 128 --imgSizeToMetaModel 84 \
#                --NWay 2 --KShot 5 --KQuery 5 \
#                --KShotTest $kShotTest --KQueryTest $kQueryTest \
#                --seed $seedval
