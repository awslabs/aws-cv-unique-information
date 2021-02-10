from abc import ABC, abstractmethod
from collections import defaultdict

from sklearn.metrics import roc_auc_score
import numpy as np
import torch

from . import utils


class Metric(ABC):
    def __init__(self, **kwargs):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError("name is not implemented")

    @abstractmethod
    def value(self, *args, **kwargs):
        raise NotImplementedError("value is not implemented")

    def on_epoch_start(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        pass

    def on_iteration_end(self, *args, **kwargs):
        pass


class Accuracy(Metric):
    """ Accuracy metric. Works in both binary and multiclass classification settings.
    """
    def __init__(self, output_key: str = 'pred', threshold=0.5, one_hot=False, **kwargs):
        """
        :param threshold: in the case of binary classification what threshold to use
        :param one_hot: whether the labels is in one-hot encoding
        """
        super(Accuracy, self).__init__(**kwargs)
        self.output_key = output_key
        self.threshold = threshold
        self.one_hot = one_hot

        # initialize and use later
        self._accuracy_storage = defaultdict(list)
        self._accuracy = defaultdict(dict)

    @property
    def name(self):
        return "accuracy"

    def value(self, epoch, partition, **kwargs):
        return self._accuracy[partition].get(epoch, None)

    def on_epoch_start(self, partition, **kwargs):
        self._accuracy_storage[partition] = []

    def on_epoch_end(self, partition, tensorboard, epoch, **kwargs):
        accuracy = np.mean(self._accuracy_storage[partition])
        self._accuracy[partition][epoch] = accuracy
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}", accuracy, epoch)

    def on_iteration_end(self, outputs, batch_labels, partition, **kwargs):
        out = outputs[self.output_key]
        if out.shape[-1] > 1:
            # multiple class
            pred = utils.to_numpy(out).argmax(axis=1).astype(np.int)
        else:
            # binary classification
            pred = utils.to_numpy(out.squeeze(dim=-1) > self.threshold).astype(np.int)
        batch_labels = utils.to_numpy(batch_labels[0]).astype(np.int)
        if self.one_hot:
            batch_labels = np.argmax(batch_labels, axis=1)
        else:
            batch_labels = batch_labels.reshape(pred.shape)
        self._accuracy_storage[partition].append((pred == batch_labels).astype(np.float).mean())


class MulticlassScalarAccuracy(Metric):
    """ Accuracy metric in case when the output is a single scalar, while num_classes > 2.
    """
    def __init__(self, output_key: str = 'pred', **kwargs):
        super(MulticlassScalarAccuracy, self).__init__(**kwargs)
        self.output_key = output_key

        # initialize and use later
        self._accuracy_storage = defaultdict(list)
        self._accuracy = defaultdict(dict)

    @property
    def name(self):
        return "accuracy"

    def value(self, epoch, partition, **kwargs):
        return self._accuracy[partition].get(epoch, None)

    def on_epoch_start(self, partition, **kwargs):
        self._accuracy_storage[partition] = []

    def on_epoch_end(self, partition, tensorboard, epoch, **kwargs):
        accuracy = np.mean(self._accuracy_storage[partition])
        self._accuracy[partition][epoch] = accuracy
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}", accuracy, epoch)

    def on_iteration_end(self, outputs, batch_labels, partition, **kwargs):
        out = outputs[self.output_key]
        assert out.shape[-1] == 1
        pred = utils.to_numpy(torch.round(out)).astype(np.int)
        batch_labels = utils.to_numpy(batch_labels[0]).astype(np.int).reshape(pred.shape)
        self._accuracy_storage[partition].append((pred == batch_labels).astype(np.float).mean())


class ROCAUC(Metric):
    """ ROC AUC for binary classification setting.
    """
    def __init__(self, output_key: str = 'pred', **kwargs):
        super(ROCAUC, self).__init__(**kwargs)
        self.output_key = output_key

        # initialize and use later
        self._score_storage = defaultdict(list)
        self._label_storage = defaultdict(list)
        self._auc = defaultdict(dict)

    @property
    def name(self):
        return "ROC AUC"

    def value(self, epoch, partition, **kwargs):
        return self._auc[partition].get(epoch, None)

    def on_epoch_start(self, partition, **kwargs):
        self._score_storage[partition] = []
        self._label_storage[partition] = []

    def on_epoch_end(self, partition, tensorboard, epoch, **kwargs):
        labels = torch.cat(self._label_storage[partition], dim=0)
        scores = torch.cat(self._score_storage[partition], dim=0)
        auc = roc_auc_score(y_true=utils.to_numpy(labels), y_score=utils.to_numpy(scores))
        self._auc[partition][epoch] = auc
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}", auc, epoch)

    def on_iteration_end(self, outputs, batch_labels, partition, **kwargs):
        pred = outputs[self.output_key]
        assert pred.shape[-1] == 1
        self._score_storage[partition].append(pred.squeeze(dim=-1))
        self._label_storage[partition].append(batch_labels[0])


class TopKAccuracy(Metric):
    def __init__(self, k, output_key: str = 'pred', **kwargs):
        super(TopKAccuracy, self).__init__(**kwargs)
        self.k = k
        self.output_key = output_key

        # initialize and use later
        self._accuracy_storage = defaultdict(list)
        self._accuracy = defaultdict(dict)

    @property
    def name(self):
        return f"top{self.k}_accuracy"

    def value(self, partition, epoch, **kwargs):
        return self._accuracy[partition].get(epoch, None)

    def on_epoch_start(self, partition, **kwargs):
        self._accuracy_storage[partition] = []

    def on_epoch_end(self, partition, tensorboard, epoch, **kwargs):
        accuracy = np.mean(self._accuracy_storage[partition])
        self._accuracy[partition][epoch] = accuracy
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}", accuracy, epoch)

    def on_iteration_end(self, outputs, batch_labels, partition, **kwargs):
        pred = utils.to_numpy(outputs[self.output_key])
        batch_labels = utils.to_numpy(batch_labels[0]).astype(np.int)

        topk_predictions = np.argsort(-pred, axis=1)[:, :self.k]
        batch_labels = batch_labels.reshape((-1, 1)).repeat(self.k, axis=1)
        topk_correctness = (np.sum(topk_predictions == batch_labels, axis=1) >= 1)

        self._accuracy_storage[partition].append(topk_correctness.astype(np.float).mean())
