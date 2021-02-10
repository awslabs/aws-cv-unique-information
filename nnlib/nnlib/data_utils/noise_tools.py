from torch.utils.data import Subset
import numpy as np


def uniform_error_confusion_matrix(error_prob, n_classes):
    cf = error_prob / (n_classes - 1) * np.ones((n_classes, n_classes))
    for i in range(n_classes):
        cf[i, i] = 1 - error_prob
    assert np.allclose(cf.sum(axis=1), 1)
    return cf


def corrupt_labels_using_confusion_matrix(dataset, confusion_matrix):
    assert isinstance(dataset, Subset)
    n_classes = confusion_matrix.shape[0]
    indices = dataset.indices
    is_corrupted = np.zeros(len(dataset), dtype=np.bool)  # 0 clean, 1 corrupted
    for current_idx, sample_idx in enumerate(indices):
        label = dataset.dataset.targets[sample_idx]
        new_label = int(np.random.choice(n_classes, 1, p=np.array(confusion_matrix[label])))
        is_corrupted[current_idx] = (label != new_label)
        dataset.dataset.targets[sample_idx] = new_label
    return is_corrupted


def get_uniform_error_corruption_fn(error_prob, n_classes):
    cf = uniform_error_confusion_matrix(error_prob=error_prob, n_classes=n_classes)

    def fn(dataset):
        return corrupt_labels_using_confusion_matrix(dataset, cf)

    return fn


def get_corruption_function_from_confusion_matrix(cf):
    def fn(dataset):
        return corrupt_labels_using_confusion_matrix(dataset, cf)

    return fn
