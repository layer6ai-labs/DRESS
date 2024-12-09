from .omniglot_loader import load_omniglot
from .celeba_loader import load_celeba_rand, load_celeba_hair, load_celeba_eyes
from .mpi3d_loader import load_mpi3d_toy, load_mpi3d_toy_hard, load_mpi3d_complex, load_mpi3d_complex_hard
from .shapes3d_loader import load_shapes3d

LOAD_DATASET = {
    'omniglot': load_omniglot,
    'celebarand': load_celeba_rand,
    'celebahair': load_celeba_hair,
    'celebaeyes': load_celeba_eyes,
    'mpi3dtoy': load_mpi3d_toy,
    'mpi3dtoyhard': load_mpi3d_toy_hard,
    'mpi3dcomplex': load_mpi3d_complex,
    'mpi3dcomplexhard': load_mpi3d_complex_hard,
    'shapes3d': load_shapes3d
}