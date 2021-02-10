from abc import ABC, abstractmethod
import os

import numpy as np

from . import utils
from .metrics import Metric
from numbers import Number
import operator


class Callback(ABC):
    def __init__(self, **kwargs):
        pass

    def on_epoch_start(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        pass

    def on_iteration_end(self, *args, **kwargs):
        pass

    @abstractmethod
    def call(self, *args, **kwargs):
        raise NotImplementedError("Function call is not implemented")


class SaveBestWithMetric(Callback):
    def __init__(self, metric, partition='val', direction='max', **kwargs):
        assert direction in ['min', 'max']
        super(SaveBestWithMetric, self).__init__(**kwargs)
        self.metric = metric
        self.partition = partition
        self.direction = direction

        if self.direction == 'max':
            self._best_result_so_far = -np.inf
        else:
            self._best_result_so_far = +np.inf

    def call(self, epoch, model, optimizer, scheduler, log_dir, **kwargs):
        result = self.metric.value(partition=self.partition, epoch=epoch)
        if result is None:
            raise ValueError(f"Metric {self.metric.name} returned none for partition "
                             f"{self.partition} at epoch {epoch}")

        update = (self.direction == 'max' and result > self._best_result_so_far) or \
                 (self.direction == 'min' and result < self._best_result_so_far)

        if update:
            self._best_result_so_far = result
            print(f"This is the best {self.partition} result w.r.t. {self.metric.name} so far. Saving the model ...")
            utils.save(model=model, optimizer=optimizer, scheduler=scheduler,
                       path=os.path.join(log_dir, 'checkpoints', f"best_{self.partition}_{self.metric.name}.mdl"))

            # save the validation result for doing model selection later
            with open(os.path.join(log_dir, f"best_{self.partition}_{self.metric.name}.txt"), 'w') as f:
                f.write(f"{result}\n")


class EarlyStoppingWithMetric(Callback):
    def __init__(self, metric, stopping_param=50, partition='val', direction='max', **kwargs):
        assert direction in ['min', 'max']
        super(EarlyStoppingWithMetric, self).__init__(**kwargs)
        self.metric = metric
        self.stopping_param = stopping_param
        self.partition = partition
        self.direction = direction

        if self.direction == 'max':
            self._best_result_so_far = -np.inf
        else:
            self._best_result_so_far = +np.inf

        self._best_result_epoch = -1

    def call(self, epoch, **kwargs) -> bool:
        """ Checks whether the training should be finished. Returning True corresponds to finishing. """
        result = self.metric.value(partition=self.partition, epoch=epoch)
        if result is None:
            raise ValueError(f"Metric {self.metric.name} returned none for partition "
                             f"{self.partition} at epoch {epoch}")

        update = (self.direction == 'max' and result > self._best_result_so_far) or \
                 (self.direction == 'min' and result < self._best_result_so_far)

        if update:
            self._best_result_so_far = result
            self._best_result_epoch = epoch

        return epoch - self._best_result_epoch > self.stopping_param


class StoppingWithOperatorApplyingOnMetric(Callback):
    def __init__(
            self,
            metric: Metric,
            metric_target_value: Number,
            partition: str,
            operator = operator.eq,
            **kwargs
    ):
        super(StoppingWithOperatorApplyingOnMetric, self).__init__(**kwargs)
        self.metric = metric
        self.metric_target_value = metric_target_value
        self.partition = partition
        self.operator = operator

    def call(self, epoch, **kwargs) -> bool:
        metric_curr_value = self.metric.value(partition=self.partition, epoch=epoch)
        return self.operator(metric_curr_value, self.metric_target_value)
