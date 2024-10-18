import os
import sys
import torch
import torch.nn as nn
from torchvision.models import resnet18
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sys.path.append("../")
from utils import *

"""
The following codes are from repo https://github.com/wuancong/FDAE:
get_last_layer_output_channels()
SeperateMaskGenerator
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
    
class SeperateMaskGenerator(nn.Module):
    def __init__(self, latent_dim, num_masks, img_size=64, use_fp16=False, channel_list=[384, 256, 128, 64],
                 num_groups=32):
        super(SeperateMaskGenerator, self).__init__()
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.init_size = img_size // 2**4 # 4 times of x2 upsample
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 384 * self.init_size ** 2))
        self.num_masks = num_masks
        self.latent_dim = latent_dim
        self.conv_blocks = nn.ModuleList()
        in_dim = 384
        for out_dim in channel_list:
            conv_block = nn.Sequential(
                nn.GroupNorm(num_groups, in_dim),  # 4
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_dim, out_dim, 3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups, out_dim),
                nn.SiLU(),  # 8
            )
            in_dim = out_dim
            self.conv_blocks.append(conv_block)
        self.conv_blocks_img = nn.Conv2d(in_dim, 1, 3, stride=1, padding=1)
        self.mask_normalize_block = nn.Softmax(dim=1)  # mask

    def forward(self, z):
        # size of input z: N x num_masks x mask_code
        # convert z from N x num_masks x mask_code to (N x num_masks) x mask_code
        N, num_masks, mask_code_dim = z.size()
        assert self.num_masks == num_masks
        assert self.latent_dim == mask_code_dim
        z = z.view(N * num_masks, mask_code_dim)
        out = self.l1(z)
        out = out.view(out.shape[0], 384, self.init_size, self.init_size)
        for block in self.conv_blocks:
            out = block(out)
        out = self.conv_blocks_img(out)
        _, _, H, W = out.size()
        out = out.view(N, num_masks, H, W)
        out = self.mask_normalize_block(out)
        return out

class ContentMaskGenerator(nn.Module):
    def __init__(self, img_size=64, semantic_group_num=2, semantic_code_dim=80, mask_code_dim=80,
                 semantic_code_adjust_dim=80,
                 use_fp16=False, encoder_type='resnet18'):
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
        if encoder_type == 'resnet18':
            self.encoder = resnet18(weights=None)
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
        #self.mask_generator = SeperateMaskGenerator(latent_dim=mask_code_dim, 
        #                                            num_masks=semantic_group_num,
        #                                            img_size=img_size)

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
        semantic_code_out = semantic_code

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

        layer_semantic_code_list = [layer_semantic_code0]

        mask_code = self.mask_decoder(features)
        mask_code = mask_code.view(mask_code.size(0), self.semantic_group_num, self.mask_code_dim)
        mask_code_out = [mask_code.view(mask_code.size(0), -1)]

        if self.semantic_group_num == 1:
            mask_code = None
            mask_code_out = []

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

        # if self.semantic_group_num == 1:
        #     mask_list = [torch.ones(x.shape[0], self.semantic_group_num, x.shape[2], x.shape[3]).cuda()]
        #     mask_output = mask_list[-1]
        # else:
        #     mask_list = [self.mask_generator(mask_code)]
        #     mask_output = mask_list[-1]

        # condition_map_list = []
        # for semantic_code_map, mask in zip(layer_semantic_code_list, mask_list):
        #     N, _, H, W = mask.size()
        #     # expand maps
        #     semantic_code_map = semantic_code_map.unsqueeze(-1).expand(
        #         N, self.semantic_code_adjust_dim, self.semantic_group_num, H, W)
        #     mask = mask.unsqueeze(1).expand_as(semantic_code_map)
        #     condition_map = torch.sum(semantic_code_map * mask, dim=2)
        #     condition_map_list.append(condition_map)
        return {'mask_code': mask_code, #'mask': mask_output,
                'semantic_code': semantic_code_list,
                #'condition_map': condition_map_list,
                #'feature': torch.cat([semantic_code_out] + mask_code_out, dim=1), 'feature_avg_pool': features
                }

 
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
        self.encoder = ContentMaskGenerator(img_size=args.imgSizeToEncoder,
                                            semantic_group_num=self.n_semantic_groups,
                                            semantic_code_dim=self._code_length,
                                            semantic_code_adjust_dim=self._code_length,
                                            mask_code_dim=self._code_length,
                                            use_fp16=True,
                                            encoder_type='resnet18')
        state_dict = torch.load(os.path.join(ENCODERDIR, f'fdae_{args.dsName}.pt'))
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
        semantic_codes = self.pca.fit_transform(semantic_codes.view(-1, self._code_length))
        mask_codes = self.pca.fit_transform(mask_codes.view(-1, self._code_length))
        semantic_codes_quantized = self.kmeans.fit_predict(semantic_codes)
        semantic_codes_quantized = np.reshape(semantic_codes_quantized, [dataset_size, self.n_semantic_groups])
        mask_codes_quantized = self.kmeans.fit_predict(mask_codes)
        mask_codes_quantized = np.reshape(mask_codes_quantized, [dataset_size, self.n_semantic_groups])
        encodings_quantized = np.concatenate([semantic_codes_quantized, mask_codes_quantized], axis=1)
        encodings_quantized = torch.from_numpy(encodings_quantized) 
        print("FDAE post_encode computed successfully!")
        return encodings_quantized
