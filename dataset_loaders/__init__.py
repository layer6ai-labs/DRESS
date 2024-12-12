from .mpi3d_loader import load_mpi3d_easy, load_mpi3d_hard
from .shapes3d_loader import load_shapes3d
from .celeba_loader import load_celeba_hair, load_celeba_eyes

LOAD_DATASET = {
    'mpi3deasy': load_mpi3d_easy,
    'mpi3dhard': load_mpi3d_hard,
    'shapes3d': load_shapes3d,
    'celebahair': load_celeba_hair,
    'celebaeyes': load_celeba_eyes
}