import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import combinations
from random import shuffle, choice
from collections import OrderedDict


class Flatten(nn.Module):
    def forward(self, target):
        return target.view(target.size(0), -1)


class NewClassifier(nn.Module):
    def __init__(self, inp=2208, h1=1024, out=102, d=0.35):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.mp = nn.AdaptiveMaxPool2d((1, 1))
        self.fla = Flatten()
        self.bn0 = nn.BatchNorm1d(inp * 2, eps=1e-05, momentum=0.1, affine=True)
        self.dropout0 = nn.Dropout(d)
        self.fc1 = nn.Linear(inp * 2, h1)
        self.bn1 = nn.BatchNorm1d(h1, eps=1e-05, momentum=0.1, affine=True)
        self.dropout1 = nn.Dropout(d)
        self.fc2 = nn.Linear(h1, out)

    def forward(self, x):
        ap = self.ap(x)
        mp = self.mp(x)
        x = torch.cat((ap, mp), dim=1)
        x = self.fla(x)
        x = self.bn0(x)
        x = self.dropout0(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        return x


class MakePairs(nn.Module):
    def __init__(self, n_samples, device=None):
        super().__init__()
        self.n_samples = n_samples
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, feature_maps, labels, phase='train', similarity=True):
        if similarity:
            x = self.similarity_matrix_pairs(feature_maps, labels, phase)
        else:
            x = self.one_shot_matrix_pairs(feature_maps,labels)
        return x

    def quadruplets_generator(self, labels, phase='train', cpu=False):
        if cpu:
            labels = labels.detach().cpu()
        indices = list(range(labels.shape[0]))

        quadruplets = [
            [[a, p], [a, n], [n, n2]]
            for a, p, n, n2 in combinations(indices, 4)
            if (labels[a] == labels[p] and
                labels[a] != labels[n] and
                labels[n2] != labels[n] and
                labels[n2] != labels[a])
        ]
        if phase == 'train':
            shuffle(quadruplets)
        return torch.tensor(quadruplets, dtype=torch.long).to(self.device)

    def similarity_matrix_pairs(self, feature_maps, labels, phase):
        # TODO: try doing this with tensor operations only
        quadruplet_indices = self.quadruplets_generator(labels, phase)
        mat_pairs = [
            torch.stack([torch.cat([feature_maps[p1], feature_maps[p2]]) for p1, p2 in quadruplet], dim=0)
            for quadruplet in quadruplet_indices
        ]
        mat_pairs = torch.stack(mat_pairs, dim=0)
        return mat_pairs  # shape = (bs,3,c,h,w), will select each type of pair with matrix_pairs[:,i,...]

    def one_shot_pair_generator(self, labels, n_samples, cpu=False):
        if cpu:
            labels = labels.detach().cpu()
        label_list = labels.reshape(-1, n_samples)[0]
        class_indices = OrderedDict(
            (label, torch.where(labels == label)[0])
            for label in label_list
        )
        pairs = []
        for index, label in enumerate(labels):
            for key in class_indices:
                support_set = class_indices[key][torch.where(class_indices[key] != index)]
                pairs.append([index, choice(support_set)])
        return torch.tensor(pairs).to(self.device)

    def one_shot_matrix_pairs(self, feature_maps, labels):
        pairs = self.one_shot_pair_generator(labels, self.n_samples)
        feature_map_pairs = [torch.cat((feature_maps[p1], feature_maps[p2])) for p1, p2 in pairs]
        feature_map_pairs = torch.stack(feature_map_pairs, dim=0)
        return feature_map_pairs


class QuadrupletsNetwork(nn.Module):
    def __init__(self, model1, model2, classifier, n_samples):
        super().__init__()
        self.model1 = model1
        self.make_pairs = MakePairs(n_samples)
        self.model2 = model2
        self.classifier = classifier
        self.softmax = nn.Softmax(dim=-1)

    # Todo make sure it works with lr_find
    def forward(self, inputs, phase='train', similarity=True):
        images, labels = inputs
        x = self.model1(images)
        pairs = self.make_pairs(x, labels, phase, similarity)
        if similarity:
            ap_features, an_features, nn2_features = pairs[:, 0, ...], pairs[:, 1, ...], pairs[:, 2, ...]
            ap_features, an_features, nn2_features = (self.model2(ap_features),
                                                      self.model2(an_features),
                                                      self.model2(nn2_features),)
            x = (self.classifier(ap_features),
                 self.classifier(an_features),
                 self.classifier(nn2_features),
                )
            x = torch.stack(x, dim=0)
        else:
            x = self.model2(pairs)
            x = self.classifier(x)
        x = self.softmax(x)
        return x