from torch.utils.data.sampler import BatchSampler
import numpy as np


class BalancedBatchSampler(BatchSampler):

    def __init__(self, labels, n_classes, n_samples):
        self.label_list = list(set(labels.numpy()))
        self.class_indices = {label: np.where(labels.numpy() == label)[0]
                              for label in self.label_list}
        for l in self.label_list:
            np.random.shuffle(self.class_indices[l])
        self.used_labels_count = {label: 0 for label in self.label_list}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            batch = []
            classes = np.random.choice(self.label_list, self.n_classes, replace=False)
            for class_ in classes:
                batch.extend(self.class_indices[class_][
                             self.used_labels_count[class_]:self.used_labels_count[class_] + self.n_samples])
                self.used_labels_count[class_] += self.n_samples
                if self.used_labels_count[class_] + self.n_samples > len(self.class_indices[class_]):
                    np.random.shuffle(self.class_indices[class_])
                    self.used_labels_count[class_] = 0
            yield batch
            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size

