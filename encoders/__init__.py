from .dino import DinoV2, Ablate_Disentangle
from .deep_cluster import DeepCluster
from .vae import *
from .fdae import FDAE, Ablate_Individual_Cluster_FDAE
from .diti import *
from .simclr_pretrain import SimCLR
from .gmvae import GMVAE
from .soda import *
from .lsd import LSD, Ablate_Align, Ablate_Individual_Cluster_LSD


def get_encoder(args, device):
    if args.encoder in ['sup', 'supall', 'supora', 'scratch']:
        return None
    if args.encoder == "dino":
        encoder = DinoV2(latent_dim=384).to(device)
    elif args.encoder == "deepcluster":
        encoder = DeepCluster(latent_dim=256, args=args).to(device)
    elif args.encoder == "vanillavae":
        encoder = VAE(latent_dim=256, args=args).to(device)
    elif args.encoder == "factorvae":
        # use the same VAE constructor function, load different trained model 
        encoder = FactorVAE(latent_dim=20, 
                            levels_per_dim=200,
                            args=args).to(device)
    elif args.encoder == "dlqvae":
        encoder = DLQVAE(latent_dim_before_quant=256,
                         latent_dim=50,
                         levels_per_dim=4,
                         args=args).to(device)
    elif args.encoder == "fdae":
        if args.dsName == "shapes3d":
            n_semantic_groups = 6
            code_length = 80
            code_length_reduced = 40
            levels_per_dim = 200
        elif args.dsName.startswith("mpi3d"):
            # probably would need to increase for mpi3d
            n_semantic_groups = 7
            code_length = 100
            code_length_reduced = 40
            levels_per_dim = 200
        elif args.dsName == "norb":
            n_semantic_groups = 8
            code_length = 80
            code_length_reduced = 40
            levels_per_dim = 200
        elif args.dsName == "causal3d":
            n_semantic_groups = 8
            code_length = 80
            code_length_reduced = 40
            levels_per_dim = 200
        else:
            print(f"FDAE unprepared for {args.dsName}!")
            exit(1)
        encoder = FDAE(n_semantic_groups=n_semantic_groups, 
                       code_length=code_length, 
                       code_length_reduced=code_length_reduced,
                       levels_per_dim=levels_per_dim,
                       args=args).to(device)
    elif args.encoder == "lsd":
        encoder = LSD(levels_per_dim=200, args=args).to(DEVICE)
    elif args.encoder == "ablate_disentangle":
        encoder = Ablate_Disentangle(latent_dim=384,
                                     levels_per_dim=200).to(DEVICE)
    elif args.encoder == "ablate_align":
        encoder = Ablate_Align(levels_per_dim=200, args=args).to(DEVICE)
    elif args.encoder == "ablate_individual_cluster":
        # FDAE config
        if args.dsName == "shapes3d":
            encoder = Ablate_Individual_Cluster_FDAE(
                        n_semantic_groups=6, 
                        code_length=80, 
                        code_length_reduced=15, # here reduced and then aggregated before clustering
                        levels_per_dim=200, # here levels_per_dim isn't used
                        args=args).to(DEVICE)
        elif args.dsName == "causal3d":
            encoder = Ablate_Individual_Cluster_FDAE(
                        n_semantic_groups=8, 
                        code_length=80, 
                        code_length_reduced=15, # here reduced and then aggregated before clustering
                        levels_per_dim=200, # here levels_per_dim isn't used
                        args=args).to(DEVICE)
        elif args.dsName == "norb":
            encoder = Ablate_Individual_Cluster_FDAE(
                        n_semantic_groups=8, 
                        code_length=80, 
                        code_length_reduced=15, # here reduced and then aggregated before clustering
                        levels_per_dim=200, # here levels_per_dim isn't used
                        args=args).to(DEVICE)
        elif args.dsName.startswith("mpi3d"):
            encoder = Ablate_Individual_Cluster_FDAE(
                        n_semantic_groups=7, 
                        code_length=100, 
                        code_length_reduced=15, # here reduced and then aggregated before clustering
                        levels_per_dim=200, # here levels_per_dim isn't used
                        args=args).to(DEVICE)
        # LSD config
        else:
            encoder = Ablate_Individual_Cluster_LSD(levels_per_dim=200, # here levels_per_dim isn't used
                                               dim_per_slot_reduced=15,
                                               args=args).to(DEVICE)
    elif args.encoder == "diti":
        encoder = DiTi(levels_per_dim=200, args=args).to(DEVICE)
    elif args.encoder == "soda":
        encoder = SODA(latent_dim=32, 
                       levels_per_dim=100,
                       args=args).to(DEVICE)
    elif args.encoder == "simclrpretrain":
        encoder = SimCLR(latent_dim=2048, args=args).to(DEVICE)
    elif args.encoder == "metagmvae":
        encoder = GMVAE(input_size=[3, args.imgSizeToEncoder, args.imgSizeToEncoder],
                        hidden_size=64,
                        component_size=args.NWay,        
                        latent_size=args.latent_dim, 
                        args=args).to(DEVICE)
    else:
        print(f"Encoder model {args.encoder} hasn't been implemented yet!")
        exit(1)  

    return encoder