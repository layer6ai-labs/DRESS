ACCURACIES_ALL = {
    'Supervised-Original': {
        'shapes3d': [66.39, 62.60, 57.70, 61.44],
        'mpi3deasy': [56.78, 57.22, 57.79, 59.23],
        'mpi3dhard': [62.11, 66.96, 63.85, 60.15],
        'smallnorb': [63.38, 60.49],
        'causal3d': [52.26, 51.68],
        'celebahair': [67.95],
        'celebaprimary': [73.26],
        'celebarand': [81.28]
    },
    'Supervised-All': {
        'shapes3d': [99.91, 99.86, 99.96, 99.98],
        'mpi3deasy': [99.86, 99.61, 99.35, 98.34],
        'mpi3dhard': [88.37, 87.38, 95.94, 92.41],
        'smallnorb': [79.16, 79.51],
        'causal3d': [86.97, 90.62],
        'celebahair': [79.74],
        'celebaprimary': [87.62],
        'celebarand': [85.33]
    },
    'Supervised-Oracle': {
        'shapes3d': [99.99, 99.96, 100.00, 99.92],
        'mpi3deasy': [100.00, 100.00, 100.00, 100.00],
        'mpi3dhard': [99.32, 99.13, 99.72, 99.50],
        'smallnorb': [81.60, 79.93],
        'causal3d': [93.37, 93.07],
        'celebahair': [87.07],
        'celebaprimary': [91.08],
        'celebarand': [90.57]
    },
    'Few-Shot Direct Adaptation': {
        'shapes3d': [63.43, 63.30, 72.79, 63.28],
        'mpi3deasy': [61.19, 60.63, 59.63, 60.90],
        'mpi3dhard': [62.57, 62.45, 61.30, 62.74],
        'smallnorb': [74.17, 75.06],
        'causal3d': [66.38, 69.80],
        'celebahair': [64.00],
        'celebaprimary': [69.12],
        'celebarand': [57.82]
    },
    'Pre-Training and Fine-Tuning': {
        'shapes3d': [65.35, 56.26, 55.77, 54.15],
        'mpi3deasy': [91.30, 93.13, 93.73, 93.54],
        'mpi3dhard': [81.98, 79.52, 78.40, 78.10],
        'smallnorb': [53.84, 58.36],
        'causal3d': [56.03],
        'celebahair': [60.01, 60.03],
        'celebaprimary': [66.49],
        'celebarand': [64.38]
    },
    'Meta-GMVAE': {
        'shapes3d': [57.04, 56.23, 58.14, 64.98],
        'mpi3deasy': [99.62, 99.64, 99.03, 99.27],
        'mpi3dhard': [49.63, 50.72, 50.31, 49.44],
        'smallnorb': [66.29, 68.86, 70.36, 68.89],
        'causal3d': [57.39, 57.88, 61.12, 60.31],
        'celebahair': [64.60, 63.54, 64.21, 64.36],
        'celebaprimary': [68.73, 67.47, 67.21, 68.07],
        'celebarand': [65.34, 65.09, 64.38, 64.95]
    },
    'PsCo':{
        'shapes3d': [98.64, 98.75, 95.93, 97.17],
        'mpi3deasy': [77.10, 83.96, 84.88, 88.15],
        'mpi3dhard': [77.55, 81.54, 78.89, 80.10],
        'smallnorb': [73.38, 75.06, 74.96, 73.31],
        'causal3d': [69.98, 70.06, 70.58, 72.46],
        'celebahair': [65.66, 66.90, 65.65, 66.76],
        'celebaprimary': [64.43, 66.29, 65.62, 67.49],
        'celebarand': [59.76, 61.30, 59.69, 61.21]
    },
    'CACTUS-DeepCluster': {
        'shapes3d': [88.03, 86.73, 84.62, 87.86],
        'mpi3deasy': [84.18, 86.27, 85.80, 83.57],
        'mpi3dhard': [71.09, 73.47, 70.88, 75.64],
        'smallnorb': [76.77, 75.18],
        'causal3d': [65.64, 65.03],
        'celebahair': [70.57],
        'celebaprimary': [71.59],
        'celebarand': [65.87]
    },
    'CACTUS-DinoV2': {
        'shapes3d': [80.27, 80.40, 81.47, 80.35],
        'mpi3deasy': [94.76, 94.95, 92.87, 94.96],
        'mpi3dhard': [82.55, 80.63, 82.47, 82.04],
        'smallnorb': [64.15],
        'causal3d': [54.09],
        'celebahair': [69.85, 69.11, 69.63, 68.88],
        'celebaprimary': [77.04, 77.48, 77.50, 76.00],
        'celebarand': [74.59, 74.42, 75.00, 73.54]
    },
    'DRESS': {
        'shapes3d': [92.54, 93.15, 93.55, 92.97],
        'mpi3deasy': [99.89, 100.00, 99.89, 100.00],
        'mpi3dhard': [83.63, 84.38, 85.58, 86.22],
        'smallnorb': [78.70, 76.96],
        'causal3d': [77.49, 75.35],
        'celebahair': [73.86, 74.18, 73.87, 73.41],
        'celebaprimary': [77.12, 77.40, 77.77, 77.33],
        'celebarand': [70.01, 67.49, 67.97, 67.63]
    },
    'Ablate-Disentangle':{
        'causal3d': [53.41, 55.24, 53.98]
    },
    'Ablate-Align': {
        'celebahair': [72.55, 73.02]
    },
    'Ablate-Individual-Cluster':{
        'celebaprimary': [73.91, 74.35]
    }
}

DIVERSITIES_ALL = {
    'Supervised-Original': {
        'shapes3d': [0.05],
        'mpi3dhard': [0.07],
        'smallnorb': [0.00],
        'causal3d': [0.02],
        'celebahair': [0.12]
    },
    'Supervised-All': {
        'shapes3d': [0.01],
        'mpi3dhard': [0.01],
        'smallnorb': [0.02],
        'causal3d': [0.01],
        'celebahair': [0.16]
    },
    'Supervised-Oracle': {
        'shapes3d': [0.01],
        'mpi3dhard': [0.03],
        'smallnorb': [0.00],
        'causal3d': [0.02],
        'celebahair': [0.17]
    },
    'CACTUS-DeepCluster': {
        'shapes3d': [0.20],
        'mpi3dhard': [0.21],
        'smallnorb': [0.32],
        'causal3d': [0.12],
        'celebahair': [0.08]
    },
    'CACTUS-DinoV2': {
        'shapes3d': [0.39],
        'mpi3dhard': [0.42],
        'smallnorb': [0.42],
        'causal3d': [0.37],
        'celebahair': [0.26]
    },
    'DRESS': {
        'shapes3d': [0.10],
        'mpi3dhard': [0.08],
        'smallnorb': [0.30],
        'causal3d': [0.27],
        'celebahair': [0.02]
    }
}