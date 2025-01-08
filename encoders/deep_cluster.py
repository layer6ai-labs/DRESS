import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

sys.path.append('../')
from utils import *

'''
AlexNet codes taken from DeepCluster repo, with top layer removed
'''
# (number of filters, kernel size, stride, pad)
CFG = {
    '2012': [(96, 11, 4, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M']
}

class AlexNet(nn.Module):
    def __init__(self, features, sobel):
        super(AlexNet, self).__init__()
        self.features = features
        # Originally classifier has a ReLU layer at the end
        # However, running deepcluster model on celeba (also trained on celebA)
        # results in features with only 3% non-zero values across all training split
        # Remove the ReLU layer at the end here (while the training was still done under 
        # the original structure with the ReLU layer)
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                            nn.Linear(256 * 6 * 6, 4096),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, 4096))

        if sobel:
            grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0, 0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1, 0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None

    def forward(self, x):
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def make_layers_features(cfg, input_dim, bn):
    layers = []
    in_channels = input_dim
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])
            if bn:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)


class DeepCluster(nn.Module):
    def __init__(self, latent_dim, args):
        super().__init__()
        self.latent_dim = latent_dim
        self.deepCluster_model = AlexNet(make_layers_features(CFG['2012'], 
                                                              input_dim=3, 
                                                              bn=True), 
                                         sobel=False)
        # wrap feature layers with nn.DataParallel to load the weights correctly
        self.deepCluster_model.features = torch.nn.DataParallel(self.deepCluster_model.features)
        # load the checkpoint from deepCluster training run
        checkpoint = torch.load(os.path.join(ENCODERDIR, f"deepcluster_{args.dsName}.pth.tar"))['state_dict']
        # remove top_layer parameters from checkpoint
        # have to cast the retured odict_keys into list to prevent "mutated during iteration" error
        lyr_keys = list(checkpoint.keys())
        for key in lyr_keys:
            if 'top_layer' in key:
                del checkpoint[key]
        self.deepCluster_model.load_state_dict(checkpoint)

        # preprocess the celebA images as required by the deepCluster alex net
        self.data_transform = None
        self.dsName = args.dsName
        if self.dsName.startswith("celeba"):
            self.data_transform = transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        self.pca = PCA(self.latent_dim)

        print(f"DeepCluster trained alex net for {args.dsName} constructed and loaded successfully!")
    
    def encode(self, input_data: torch.Tensor) -> torch.Tensor:
        assert input_data.size(1) == 3, "deepcluster only takes input with 3 channels!"
        if self.dsName.startswith("celeba"):
            # for now only do the color channel normalization on celeba
            # this should match what's done in training the deepcluster models
            input_data = self.data_transform(input_data)
        with torch.no_grad():
            encodings = self.deepCluster_model(input_data)
        assert encodings.size(1) == 4096  
        return encodings
        
    # The input should be on CPU 
    def post_encode(self, encodings):
        encodings = self.pca.fit_transform(encodings)
        print("DeepCluster post_encode computed successfully!")
        return torch.from_numpy(encodings.astype(np.float32))