import os
import sys
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sys.path.append("../")
from utils import *

"""
The following codes are from repo https://github.com/wuancong/FDAE:
get_last_layer_output_channels()
ContentMaskGenerator
"""

def get_last_layer_output_channels(model):
    # Get the last layer of the model
    last_layer = list(model.modules())[-1]
    # Check if the last layer is a linear or convolutional layer
    if isinstance(last_layer, nn.Linear):
        # For a linear layer, return the number of output features
        return last_layer.in_features
    elif isinstance(last_layer, nn.Conv2d):
        # For a convolutional layer, return the number of output channels
        return last_layer.out_channels
    elif isinstance(last_layer, nn.BatchNorm2d):
        return last_layer.num_features
    elif isinstance(last_layer, nn.AdaptiveAvgPool2d):
        return get_last_layer_output_channels(torch.nn.Sequential(*list(model.children())[:-1]))
    else:
        raise ValueError('last_layer value not supported')


class ContentMaskGenerator(nn.Module):
    def __init__(self, semantic_group_num, semantic_code_dim, 
                 semantic_code_adjust_dim, mask_code_dim,
                 use_fp16, encoder_type):
        '''
        semantic_group_num: concept number N
        semantic_code_dim: dimensionality of content codes
        mask_code_dim: dimensionality of mask codes
        semantic_code_adjust_dim: dimensionality of content codes after dimension adjustment
        '''
        super(ContentMaskGenerator, self).__init__()
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.semantic_group_num = semantic_group_num
        self.semantic_code_adjust_dim = semantic_code_adjust_dim
        self.semantic_code_dim = semantic_code_dim
        self.mask_code_dim = mask_code_dim
        # for resnet, don't use pretrained weights, therefore no
        # preprocessing (resizing and normalization) done within the encoder
        if encoder_type == 'resnet18':
            self.encoder = resnet18(weights=None)
        elif encoder_type == 'resnet50':
            self.encoder = resnet50(weights=None)
        else:
            raise(ValueError('Unsupported encoder type'))
        encoder_out_dim = get_last_layer_output_channels(self.encoder)
        self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-1])
        self.gn_dim = 32

        self.semantic_decoder1 = nn.Linear(encoder_out_dim, self.semantic_code_dim * self.semantic_group_num)
        self.semantic_code_dim = semantic_code_dim

        self.semantic_decoder2 = nn.ModuleList()
        for i in range(self.semantic_group_num):
            self.semantic_decoder2.append(nn.Linear(self.semantic_code_dim, semantic_code_adjust_dim))

        self.mask_decoder = nn.Linear(encoder_out_dim, mask_code_dim * semantic_group_num)

    def forward(self, x, model_kwargs=None):
        swap_info = None  # swap_info is used for image editing by swapping latent codes
        if model_kwargs is not None:
            swap_info = model_kwargs.get('swap_info')
        if swap_info is not None:
            source_ind = swap_info['source_ind']
            target_ind = swap_info['target_ind']
            semantic_group = swap_info['semantic_group']
            mask_group = swap_info['mask_group']

        x = x.type(self.dtype)

        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        semantic_code = self.semantic_decoder1(features)
        semantic_code_list = semantic_code.chunk(self.semantic_group_num, dim=1)
        layer_semantic_code0_list = [net(code)  # code nxd_group
                                     for code, net in zip(semantic_code_list, self.semantic_decoder2)]
        layer_semantic_code0 = torch.stack(layer_semantic_code0_list, dim=2).unsqueeze(3)  # nxdxgroup_numx1

        if swap_info is not None:
            # layer_semantic_code0: nxdxgroup_numx1
            layer_semantic_code0_new = torch.zeros((len(source_ind), len(target_ind), layer_semantic_code0.shape[1],
                                                    layer_semantic_code0.shape[2], layer_semantic_code0.shape[3]),
                                                   dtype=layer_semantic_code0.dtype).cuda()
            swap_array = swap_info.get('cluster_center')
            for i_s, s_ind in enumerate(source_ind):
                for i_t, t_ind in enumerate(target_ind):
                    layer_semantic_code0_new[i_s, i_t] = layer_semantic_code0[s_ind]
                    #swap
                    for g in semantic_group:
                        if swap_array is None:
                            layer_semantic_code0_new[i_s, i_t, :, g, :] = layer_semantic_code0[t_ind, :, g, :]
                        else:
                            layer_semantic_code0_new[i_s, i_t, :, g, :] = torch.from_numpy(swap_array['semantic'][g][:, i_t]).unsqueeze(1).cuda().half()
            layer_semantic_code0 = layer_semantic_code0_new.reshape(len(source_ind) * len(target_ind), layer_semantic_code0.shape[1],
                                               layer_semantic_code0.shape[2], layer_semantic_code0.shape[3])


        mask_code = self.mask_decoder(features)
        mask_code = mask_code.view(mask_code.size(0), self.semantic_group_num, self.mask_code_dim)

        if self.semantic_group_num == 1:
            mask_code = None

        if swap_info is not None:
            # mask_code nxgroup_numxd
            mask_code_new = torch.zeros((len(source_ind), len(target_ind), self.semantic_group_num, self.mask_code_dim),
                                        dtype=mask_code.dtype).cuda()
            swap_array = swap_info.get('cluster_center')
            for i_s, s_ind in enumerate(source_ind):
                for i_t, t_ind in enumerate(target_ind):
                    mask_code_new[i_s, i_t] = mask_code[s_ind]
                    #swap
                    for g in mask_group:
                        if swap_array is None:
                            mask_code_new[i_s, i_t, g, :] = mask_code[t_ind, g, :]
                        else:
                            mask_code_new[i_s, i_t, g, :] = torch.from_numpy(swap_array['mask'][g][:, i_t]).unsqueeze(1).cuda().half()
            mask_code = mask_code_new.reshape(len(source_ind) * len(target_ind), self.semantic_group_num, self.mask_code_dim)

        return {'mask_code': mask_code, 
                'semantic_code': semantic_code_list}

 
class FDAE(nn.Module):
    def __init__(self, 
                 n_semantic_groups, 
                 code_length, 
                 code_length_reduced, 
                 levels_per_dim,
                 args):
        super(FDAE, self).__init__()
        self.n_semantic_groups = n_semantic_groups
        # For each group, we treat the semantic and mask codes as 
        # separate disentangled dimension
        self.latent_dim = self.n_semantic_groups * 2
        self._code_length = code_length
        self._code_length_reduced = code_length_reduced
        self._levels_per_dim = levels_per_dim
        self.encoder = ContentMaskGenerator(semantic_group_num=self.n_semantic_groups,
                                            semantic_code_dim=self._code_length,
                                            semantic_code_adjust_dim=self._code_length,
                                            mask_code_dim=self._code_length,
                                            use_fp16=True,
                                            encoder_type='resnet18')
        if args.dsName.startswith("mpi3d"):
            dsName_base = "mpi3d"
        elif args.dsName.startswith("celeba"):
            dsName_base = "celeba"
        else:
            dsName_base = args.dsName
        state_dict = torch.load(os.path.join(ENCODERDIR, f'fdae_{dsName_base}.pt'))
        # With strict=False, other components (diffusion model) are not loaded 
        # for the condition generator
        self.encoder.load_state_dict(state_dict, strict=False)
        self.encoder.half()
        self.encoder.to(DEVICE)
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        self.img_size = args.imgSizeToEncoder

        # for post processing into discrete codes
        self.pca = PCA(self._code_length_reduced)
        self.kmeans = KMeans(n_clusters=self._levels_per_dim, 
                             init='k-means++', 
                             n_init=1, 
                             max_iter=100)

    def encode(self, input_data: torch.Tensor) -> torch.Tensor:
        batch_size = input_data.shape[0]
        assert input_data.shape == (batch_size, 3, self.img_size, self.img_size), \
                    f"Incorrect input image shape for FDAE: {input_data.shape}"
        # Original code from FDAE normalizing images to range -1 ~ 1  
        # transforms.ToTensor normalizes images to range 0 ~ 1
        # convert them here
        assert torch.min(input_data) >= 0, "ToTensor() operator should results in normalized images with min pixel not less than 0"
        input_data = input_data * 2 - 1

        with torch.no_grad():
            encodings_outputs = self.encoder.forward(input_data)
        
        semantic_codes, mask_codes = encodings_outputs['semantic_code'], encodings_outputs['mask_code']
        # reorganize semantic codes from original tuple list
        semantic_codes = torch.stack(semantic_codes, dim=1)
        assert semantic_codes.shape == mask_codes.shape == (batch_size, self.n_semantic_groups, self._code_length)

        return torch.stack([semantic_codes, mask_codes], dim=2) 

    def post_encode(self, encodings):
        print("FDAE start post encode...")
        dataset_size = encodings.shape[0]
        assert encodings.shape == (dataset_size, self.n_semantic_groups, 2, self._code_length)
        # first, use PCA to reduce the dimension of each code 
        # do PCA separately in the semantic code and mask code space
        semantic_codes, mask_codes = encodings.split(1, dim=2)
        semantic_codes = semantic_codes.view(-1, self._code_length)
        mask_codes = mask_codes.view(-1, self._code_length)
        if self._code_length_reduced < self._code_length:
            semantic_codes = self.pca.fit_transform(semantic_codes)
            mask_codes = self.pca.fit_transform(mask_codes)            
        semantic_codes_quantized = self.kmeans.fit_predict(semantic_codes)
        semantic_codes_quantized = np.reshape(semantic_codes_quantized, [dataset_size, self.n_semantic_groups])
        mask_codes_quantized = self.kmeans.fit_predict(mask_codes)
        mask_codes_quantized = np.reshape(mask_codes_quantized, [dataset_size, self.n_semantic_groups])
        encodings_quantized = np.concatenate([semantic_codes_quantized, mask_codes_quantized], axis=1)
        encodings_quantized = torch.from_numpy(encodings_quantized) 
        print("FDAE post_encode computed successfully!")
        return encodings_quantized
