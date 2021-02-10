import os
import pickle

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import numpy as np
from PIL import Image
import pandas as pd
import tqdm


class Birds(VisionDataset):
    """`Birds <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Citation
    @techreport{WahCUB_200_2011,
        Title = {{The Caltech-UCSD Birds-200-2011 Dataset}},
        Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
        Year = {2011}
        Institution = {California Institute of Technology},
        Number = {CNS-TR-2011-001}
    }

    Args:
        root (string): Root directory of dataset where directory
            ``CUB_200_2011`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = 'CUB_200_2011'
    url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"

    train_data_file = "train_data.npz"
    test_data_file = "test_data.npz"

    meta_data_file = "meta.p"

    relevant_files = [train_data_file, test_data_file, meta_data_file]

    def __init__(self, root, train: bool, transform=None, target_transform=None, download: bool = False):
        super(Birds, self).__init__(root, transform=transform,
                                           target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.train_data_file
        else:
            data_file = self.test_data_file

        # now load the pickled numpy arrays
        with np.load(os.path.join(self.root, self.base_folder, data_file), allow_pickle=True) as f:
            self.data = f['data']
            self.targets = f['labels']

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta_data_file)
        if not check_integrity(path):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile)
            self.classes = data["classes"]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for file_name in self.relevant_files:
            file_path = os.path.join(root, self.base_folder, file_name)
            if not check_integrity(file_path):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename)

        train_test_split_file = os.path.join(self.root, self.base_folder, "train_test_split.txt")
        image_labels_file = os.path.join(self.root, self.base_folder, "image_class_labels.txt")
        classes_file = os.path.join(self.root, self.base_folder, "classes.txt")
        images_file = os.path.join(self.root, self.base_folder, "images.txt")
        # bbox_file = os.path.join(self.root, self.base_folder, "bounding_boxes.txt")

        meta_file = os.path.join(self.root, self.base_folder, self.meta_data_file)

        # get labels, image ids, and names
        is_training_df = pd.read_csv(train_test_split_file, delimiter=" ", header=None,
                                     names=["image_id", "is_training_image"])
        labels_df = pd.read_csv(image_labels_file, delimiter=" ", header=None, names=["image_id", "class_id"])
        images_df = pd.read_csv(images_file, delimiter=" ", header=None, names=["image_id", "image_file"])

        # bbox_df = pd.read_csv(bbox_file, delimiter=" ", header=None, names=["image_id", "x", "y", "width", "height"])

        data_df = is_training_df.merge(labels_df, on="image_id").merge(images_df, on="image_id")
        # .merge(bbox_df, on="image_id")

        train_data = []
        train_labels = []

        test_data = []
        test_labels = []

        for _, row in tqdm.tqdm(data_df.iterrows(), total=len(data_df), desc="parsing birds data"):
            # get class label
            label = row["class_id"] - 1  # since the original dataset was 1 indexed
            raw_image = Image.open(os.path.join(self.root, self.base_folder, "images", row["image_file"]))
            rgb_image = raw_image.convert('RGB')
            image = np.array(rgb_image)

            if row["is_training_image"]:
                train_data.append(image)
                train_labels.append(label)
            else:
                test_data.append(image)
                test_labels.append(label)

        train_data = np.array(train_data)
        train_labels = np.array(train_labels)

        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        # postprocess files to generate train and test npz files, as well as a meta file containing information
        np.savez(os.path.join(self.root, self.base_folder, self.train_data_file), data=train_data, labels=train_labels)

        np.savez(os.path.join(self.root, self.base_folder, self.test_data_file), data=test_data, labels=test_labels)

        # load in the classes via pandas, then convert to list
        classes = pd.read_csv(classes_file, delimiter=" ", header=None, index_col=0)[1].to_list()
        meta_data = {"classes": classes}

        with open(meta_file, "wb") as f:
            pickle.dump(meta_data, f)