from .mpi3d_loader import load_mpi3d_easy, load_mpi3d_hard
from .shapes3d_loader import load_shapes3d
from .celeba_loader import *
from .norb_loader import load_norb
from .causal3d_loader import load_causal3d

LOAD_DATASET = {
    'mpi3deasy': load_mpi3d_easy,
    'mpi3dhard': load_mpi3d_hard,
    'shapes3d': load_shapes3d,
    'celebarand': load_celeba_rand,
    'celebahair': load_celeba_hair,
    'celebaprimary': load_celeba_primary,
    'norb': load_norb,
    'causal3d': load_causal3d,
}