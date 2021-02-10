import numpy as np
import torch.utils.data


class TwoAugmentationWrapper(torch.utils.data.Dataset):
    """ Takes a dataset instance and creates another dataset which returns two data augmentations per example.
    """
    def __init__(self, dataset):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        """
        super(TwoAugmentationWrapper, self).__init__()
        self.dataset = dataset
        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # generate two views from one example and return both as a single sample
        (x1, y) = self.dataset[idx]
        (x2, y) = self.dataset[idx]
        return [x1, x2], y


class SemiSupervisedWrapper(torch.utils.data.Dataset):
    """ Takes a dataset and creates a semi-supervised dataset.
    :NOTE: the "label" for unlabeled examples will be -1.
    """
    def __init__(self, dataset, labeled_count_per_class):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        :param labeled_count_per_class: number of labeled examples per class
        """
        self.dataset = dataset
        self.labeled_count_per_class = labeled_count_per_class

        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

        # mark unlabeled examples
        self.is_labeled = np.zeros(len(dataset), dtype=np.bool)
        all_labels = [y for (x, y) in self.dataset]
        all_labels = np.array(all_labels)
        num_classes = np.max(all_labels)

        for y in range(num_classes):
            cur_label_indices = np.where(all_labels == y)[0]
            choose_cnt = min(labeled_count_per_class, len(cur_label_indices))
            labeled_indices = np.random.choice(cur_label_indices, size=choose_cnt, replace=False)
            for idx in labeled_indices:
                self.is_labeled[idx] = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.is_labeled[idx]:
            return x, y
        return x, -1
