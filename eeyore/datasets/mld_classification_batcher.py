# Minimum likelihood distance batcher for classification

import copy
import functools
import operator
import random
import torch

from .mld_batcher import MLDBatcher

class MLDClassificationBatcher(MLDBatcher):
    def __init__(self, num_batches, chunk_sizes, dataset=None):
        self.num_batches = num_batches

        self.chunk_sizes = chunk_sizes
        assert len(self.chunk_sizes) == 2

        self.set_dataset(dataset)

    def set_dataset(self, dataset):
        self.dataset = dataset

        if self.dataset is not None:
            self.num_points = len(dataset)
            self.num_classes = len(dataset.y[0])

            label_argmax = torch.argmax(self.dataset.y, axis=1)

            self.class_indices = [[] for _ in range(self.num_classes)]
            for i in range(self.num_points):
                self.class_indices[label_argmax[i].item()].append(i)

            self.class_props = [len(self.class_indices[i]) / self.num_points for i in range(self.num_classes)]

            self.class_num_batch_points = [
                [int(self.class_props[j]*self.chunk_sizes[i]) for j in range(self.num_classes)] for i in range(2)
            ]

    def batch_size(self):
        return sum(self.chunk_sizes)

    def fill_class_sizes(self):
        class_num_batch_points = copy.deepcopy(self.class_num_batch_points)

        sampled_classes = [
            random.choices(range(self.num_classes), k=self.chunk_sizes[i]-sum(class_num_batch_points[i])) for i in range(2)
        ]

        for i in range(2):
            for j in sampled_classes[i]:
                class_num_batch_points[i][j] = class_num_batch_points[i][j] + 1

        return class_num_batch_points

    def get_batch(self, model, params, fill=True):
        class_num_batch_points = [self.fill_class_sizes() for _ in range(self.num_batches)]

        mld_distance = float('inf')

        for i in range(self.num_batches):
            indices = []

            indices.extend([
                random.sample(self.class_indices[j], class_num_batch_points[i][0][j]) for j in range(self.num_classes)
            ])

            rest_indices = [list(set(self.class_indices[j]) - set(indices[j])) for j in range(self.num_classes)]

            indices.extend(
                [random.sample(rest_indices[j], class_num_batch_points[i][1][j]) for j in range(self.num_classes)]
            )

            indices = functools.reduce(operator.iconcat, indices, [])

            indices.sort()

            distance = 0.

            for j in range(2):
                log_lik_vals = model.set_params_and_log_lik(params[j].clone().detach(), self.dataset.x, self.dataset.y)

                distance = distance + (log_lik_vals.mean() - log_lik_vals[indices].mean()).abs()

            distance = distance.sqrt().item()

            if distance < mld_distance:
                mld_indices = indices
                mld_distance = distance

        return self.dataset.x[mld_indices, :], self.dataset.y[mld_indices, :]
