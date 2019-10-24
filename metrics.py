# from abc import ABC, abstractmethod
#
#
# class Metric(ABC):
#     def __init__(self):
#         pass
#
#     @abstractmethod
#     def on_epoch_begin(self, **kwargs):
#         pass
#
#     def on_batch_begin(self, **kwargs):
#         pass
#
#     @abstractmethod
#     def on_batch_end(self, output, target, **kwargs):
#         pass
#
#     @abstractmethod
#     def on_epoch_end(self):
#         pass
import torch


class Accuracy:
    def __init__(self):
        self.total, self.count = 0., 0

    def on_epoch_begin(self, **kwargs):
        self.total, self.count = 0., 0

    def on_batch_end(self, output, labels, n_samples):
        output = output[..., 1].view(-1, n_samples)
        label_list = labels.reshape(-1, n_samples)[:, 0]
        max_similarities = output.argmin(dim=-1)
        pred = label_list[max_similarities]
        result = pred == labels
        self.total += result[result].size(0)
        self.count += labels.size(0)

    def on_epoch_end(self):
        return self.total / self.count














