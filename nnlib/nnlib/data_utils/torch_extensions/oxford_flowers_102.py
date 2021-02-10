import os
import shutil

import scipy.io
from torchvision import datasets
from torchvision.datasets.utils import download_and_extract_archive, download_url


class OxfordFlowers102(datasets.ImageFolder):
    """`Oxford-flowers-102 <http://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.
    Citation
    @InProceedings{Nilsback08,
      author       = "Maria-Elena Nilsback and Andrew Zisserman",
      title        = "Automated Flower Classification over a Large Number of Classes",
      booktitle    = "Indian Conference on Computer Vision, Graphics and Image Processing",
      month        = "Dec",
      year         = "2008",
    }

    Args:
        root (string): Root directory of dataset where directory
            ``oxford-flowers-102`` exists or will be saved to if download is set to True.
        split (str, optional): 'train', 'val', or 'test'.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    images_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    labels_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    splits_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

    raw_ims_file = "102flowers.tgz"
    label_file = "imagelabels.mat"
    splits_file = "setid.mat"

    jpg_dir = "jpg"

    relevant_files = ['train', 'val', 'test']

    def __init__(self, root, split: str, transform=None, target_transform=None, download: bool = False,
                 cleanup: bool = True):
        assert split in ['train', 'val', 'test']
        self.split = split  # training set or test set
        self.root = root
        if download:
            self.download(cleanup)

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        data_root = os.path.join(root, self.split)
        super(OxfordFlowers102, self).__init__(data_root, transform=transform, target_transform=target_transform)

    def _check_integrity(self):
        for file_name in self.relevant_files:
            file_path = os.path.join(self.root, file_name)
            if not os.path.exists(file_path):
                return False
        return True

    def download(self, cleanup):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_and_extract_archive(self.images_url, os.path.join(self.root, "raw_data"), filename=self.raw_ims_file)
        download_url(self.labels_url, os.path.join(self.root, "raw_data"), self.label_file)
        download_url(self.splits_url, os.path.join(self.root, "raw_data"), self.splits_file)

        mat = scipy.io.loadmat(os.path.join(self.root, 'raw_data', self.label_file))
        labels = mat['labels'][0]
        classes = [str(x) for x in set(list(labels))]

        mat = scipy.io.loadmat(os.path.join(self.root, 'raw_data', self.splits_file))
        train = mat['trnid'][0]
        val = mat['valid'][0]
        test = mat['tstid'][0]

        def create_split(split_name, split_indices):
            os.mkdir(os.path.join(self.root, split_name))
            for label_name in classes:
                os.mkdir(os.path.join(self.root, split_name, label_name))
            for sample_idx in split_indices:
                file_name = f'image_{sample_idx:05d}.jpg'
                src = os.path.join(self.root, 'raw_data', 'jpg', file_name)
                dest = os.path.join(self.root, split_name, str(labels[sample_idx - 1]), file_name)
                shutil.copy(src, dest)

        create_split('train', train)
        create_split('val', val)
        create_split('test', test)

        if cleanup:
            shutil.rmtree(os.path.join(self.root, "raw_data"))
