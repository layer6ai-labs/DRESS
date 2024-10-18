# reference: https://github.com/mahayat/SimCLR-2/blob/master/models/resnet_simclr.py
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SimCLR(nn.Module):
    """
    Build a SimCLR model.
    """
    def __init__(self, latent_dim, args):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimCLR, self).__init__()

        # create the encoder
        model =  torchvision.models.__dict__[args.backbone](pretrained=False, 
                                                        num_classes=latent_dim, 
                                                        zero_init_residual=True)
        # add projection head
        self.feature_dim =  model.fc.in_features

        self.backbone = nn.Sequential(*list(model.children())[:-1])
        # projection MLP
        self.projector = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim), 
        nn.ReLU(), nn.Linear(self.feature_dim, latent_dim))


    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            z1, z2
        """

        # compute features for one view
        h1 = self.backbone(x1).squeeze() # NxC
        h2 = self.backbone(x2).squeeze() # NxC

        z1 = self.projector(h1) # NxC
        z2 = self.projector(h2) # NxC

        return z1, z2

    def get_representations(self, x):
        """
        Input:
            x: images
        Output:
            z: features
        """
        return self.backbone(x).squeeze()

    def info_nce_loss(self, z1, z2, device, temperature=0.07):
        features = torch.cat([z1, z2], dim=0)
        bs = z1.shape[0]
        n_views = 2
        labels = torch.cat([torch.arange(bs) for _ in range(n_views)], dim=0)
        
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        assert similarity_matrix.shape == (n_views * bs, n_views * bs)
        assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / temperature
        return logits, labels

