from .mpi3d_loader import load_mpi3d_easy, load_mpi3d_hard
from .shapes3d_loader import load_shapes3d
from .celeba_loader import load_celeba_hair, load_celeba_notable
from .norb_loader import load_norb
from .causal3d_loader import load_causal3d
from .birds_loader import load_birds

LOAD_DATASET = {
    'mpi3deasy': load_mpi3d_easy,
    'mpi3dhard': load_mpi3d_hard,
    'shapes3d': load_shapes3d,
    'celebahair': load_celeba_hair,
    'celebanotable': load_celeba_notable,
    'norb': load_norb,
    'causal3d': load_causal3d,
    'birds': load_birds
}