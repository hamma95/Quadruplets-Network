import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm_notebook as tqdm

import pandas as pd
import copy
import time
from IPython.display import display, clear_output
from collections import OrderedDict


def flatten_model(network, all_layers):
    for layer in network.children():
        if not list(layer.children()):  # if leaf node, add it to list
            all_layers.append(layer)
        else:
            flatten_model(layer, all_layers)
    return all_layers


class Epoch:
    def __init__(self):
        self.number = 0
        self.loss = 0
        self.start_time = None
        self.duration = None
        self.len = 0


class Learner:
    def __init__(self,
                 train_dl,
                 val_dl,
                 samples_per_class,
                 model,
                 loss_func,
                 optimizer,
                 scheduler,
                 metrics=None,
                 device=None,
                 params=None, ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataloaders = {'train': train_dl,
                            'val': val_dl}
        self.samples_per_class = samples_per_class
        self.loss_func = loss_func
        self.metrics = metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = Epoch()
        self.tb = SummaryWriter(comment=f'-{params}')
        self.best_acc = 0
        self.chosen_metric = metrics[0] if self.metrics else None
        self.best_model_wts = copy.deepcopy(self.model.state_dict)
        self.results = OrderedDict({'epoch': [],
                                    'train loss': [],
                                    'val loss': [],
                                    **{f'{metric.__class__.__name__}': [] for metric in self.metrics},
                                    'epoch duration': []})

    def show_batch_tb(self, phase):
        images, labels = next(iter(self.dataloaders[phase]))
        grid = torchvision.utils.make_grid(images)
        self.tb.add_image('images', grid)
        self.tb.add_graph(self.model, [images.to(self.device), labels.to(self.device)])

    def begin_run(self):
        for key in self.results:
            self.results[key] = []

    def begin_epoch(self, phase):
        if phase == 'train':
            self.epoch.start_time = time.time()
        self.epoch.loss = 0
        self.epoch.len = 0

    def end_epoch(self, phase):
        loss = self.epoch.loss / self.epoch.len
        self.tb.add_scalar(f'{phase}Loss', loss, self.epoch.number)
        self.results[f'{phase} loss'].append(loss)

        if phase == 'val':
            self.epoch.number += 1
            self.results["epoch"].append(self.epoch.number)
            self.epoch.duration = time.time() - self.epoch.start_time
            self.results['epoch duration'].append(self.epoch.duration)
            for metric in self.metrics:
                last_acc = metric.on_epoch_end()
                self.tb.add_scalar(f'{metric.__class__.__name__}', last_acc, self.epoch.number)
                self.results[f'{metric.__class__.__name__}'].append(last_acc)
            df = pd.DataFrame.from_dict(self.results, orient='columns')
            clear_output(wait=True)
            display(df)
            if self.chosen_metric:
                acc_value = self.chosen_metric.on_epoch_end()
                if acc_value > self.best_acc:
                    self.best_acc = acc_value
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def track_loss(self, loss, n_samples):
        self.epoch.loss += loss.item() * n_samples
        self.epoch.len += n_samples

    def fit(self, num_epochs):
        self.begin_run()
        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                self.begin_epoch(phase)
                if phase == 'train':
                    self.model.train()  # Set self.model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode
                # Iterate over data.
                n_iterations = len(self.dataloaders[phase])
                iterator = iter(self.dataloaders[phase])
                for _ in tqdm(range(n_iterations)):
                    inputs, labels = next(iterator)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model([inputs, labels], phase)
                        loss, n_samples = self.loss_func(outputs)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    # statistics
                    self.track_loss(loss, n_samples)
                    if phase == 'val':
                        for metric in self.metrics:
                            outputs = self.model([inputs, labels], phase, similarity=False)
                            metric.on_batch_end(outputs, labels, self.samples_per_class)
                    if phase == 'train':
                        if self.scheduler is not None:
                            self.scheduler.step()
                self.end_epoch(phase)
        self.tb.close()

    def freeze_block(self, index):
        frozen, unfrozen = list(self.model.children())[:index], list(self.model.children())[index:]
        for frozen_block, unfrozen_block in zip(frozen, unfrozen):
            for param1, param2 in zip(frozen_block.parameters(), unfrozen_block.parameters()):
                param1.requires_grad = False
                param2.requires_grad = True

    def freeze(self):
        self.freeze_block(-1)

    def unfreeze(self):
        self.freeze_to_layer(0)

    def freeze_to_layer(self, index):
        all_layers = flatten_model(self.model, [])
        frozen, unfrozen = all_layers[:index], all_layers[index:]
        for frozen_layer, unfrozen_layer in zip(frozen, unfrozen):
            for param1, param2 in zip(frozen_layer.parameters(), unfrozen_layer.parameters()):
                param1.requires_grad = False
                param2.requires_grad = True

