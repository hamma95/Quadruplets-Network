import torch
import torch.nn as nn
import torch.nn.functional as F


class QuadrupletsLoss(nn.Module):
    def __init__(self, mOHNM=True):
        super(QuadrupletsLoss, self).__init__()
        self.w1 = 1
        self.w2 = 0.5
        self.alpha1 = self.w1
        self.alpha2 = self.w2
        self.mOHNM = mOHNM

    def forward(self, output, *args):
        distances = output[..., 1]  # selecting the second element in the last softmax layer as indicator for distance
        if self.mOHNM:
            self.alpha1, self.alpha2 = self.mohnm(*distances)
        chosen_ap, chosen_an, chosen_nn2 = self.choose_quadruplets(*distances)

        losses = (F.relu(chosen_ap**2 - chosen_an**2 + self.alpha1) +
                  F.relu(chosen_ap**2 - chosen_nn2**2 + self.alpha2))
        return losses.mean(), chosen_ap.size(0)

    def choose_quadruplets(self, anchor_positives, anchor_negatives, negative_negative2):
        condition1 = (anchor_positives**2 - anchor_negatives**2) < max(self.alpha1, 0)
        condition2 = (anchor_positives**2 - negative_negative2**2) < max(self.alpha2, 0)
        chosen_indices = torch.where(condition1 & condition2)[0]
        [chosen_ap, chosen_an, chosen_nn2] = [anchor_positives[chosen_indices],
                                              anchor_negatives[chosen_indices],
                                              negative_negative2[chosen_indices]
                                              ]
        return chosen_ap, chosen_an, chosen_nn2

    def mohnm(self, anchor_positives, anchor_negatives, negative_negative2):
        negatives = torch.cat([anchor_negatives, negative_negative2])
        alpha1 = self.w1*(torch.mean(negatives**2) - torch.mean(anchor_positives**2))
        alpha2 = (self.w2 / self.w1) * alpha1
        return alpha1, alpha2

