import torch
from itertools import permutations
from random import shuffle, choice
import numpy as np
from collections import OrderedDict


# def quadruplets_generator(labels, phase='train', cpu=True):
#     if cpu:
#         labels = labels.detach().cpu()
#     indices = list(range(labels.shape[0]))
#
#     quadruplets = [
#         [[a, p], [a, n], [n, n2]]
#         for a, p, n, n2 in permutations(indices, 4)
#         if (labels[a] == labels[p] and
#             labels[a] != labels[n] and
#             labels[n2] != labels[n] and
#             labels[n2] != labels[a])
#     ]
#     if phase == 'train':
#         shuffle(quadruplets)
#     return torch.tensor(quadruplets, dtype=torch.long)
#
#
# def matrix_pairs(feature_maps, quadruplet_indices, device=None):
#     # TODO: try doing this with tensor operations only
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     mat_pairs = [
#         [torch.cat([feature_maps[p1], feature_maps[p2]]) for pairs in quadruplet
#          for p1, p2 in pairs]
#         for quadruplet in quadruplet_indices
#     ]
#     return torch.tensor(mat_pairs).to(
#         device)  # shape = (bs,3,c,h,w), will select each type of pair with matrix_pairs[:,i,...]
#

def flatten_model(network, all_layers):
    for layer in network.children():
        if not list(layer.children()):  # if leaf node, add it to list
            all_layers.append(layer)
        else:
            flatten_model(layer, all_layers)
    return all_layers


# def one_shot_pair_generator(labels, n_samples):
#
#     labels = labels.detach().cpu().numpy()
#     label_list = labels.reshape(-1, n_samples)[0]
#     class_indices = OrderedDict(
#         (label, np.where(labels == label)[0])
#         for label in label_list
#     )
#     pairs = []
#     for index, label in enumerate(labels):
#         for key in class_indices:
#             support_set = class_indices[key][np.where(class_indices[key] != index)]
#             pairs.extend([index, choice(support_set)])
#     return pairs
#
#
# def one_shot_feature_pairs(feature_maps, labels, n_samples,  device=None):
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     pairs = one_shot_pair_generator(labels, n_samples)
#     feature_map_pairs = [torch.cat((feature_maps[p1], feature_maps[p2])) for p1, p2 in pairs]
#     return torch.tensor(feature_map_pairs).to(device)
#




