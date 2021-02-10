import os
import shutil

import tqdm
import scipy.io
from PIL import Image
from torchvision import datasets
from torchvision.datasets.utils import download_and_extract_archive, download_url


class Cars(datasets.ImageFolder):
    """`Cars <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Citation
    @inproceedings{KrauseStarkDengFei-Fei_3DRR2013,
      title = {3D Object Representations for Fine-Grained Categorization},
      booktitle = {4th International IEEE Workshop on  3D Representation and Recognition (3dRR-13)},
      year = {2013},
      address = {Sydney, Australia},
      author = {Jonathan Krause and Michael Stark and Jia Deng and Li Fei-Fei}
    }

    Args:
        root (string): Root directory of dataset where directory
            ``car196`` exists or will be saved to if download is set to True.
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

    ims_url = "http://imagenet.stanford.edu/internal/car196/car_ims.tgz"
    annos_url = "http://imagenet.stanford.edu/internal/car196/cars_annos.mat"

    raw_ims_file = "car_ims.tgz"
    raw_annos_file = "cars_annos.mat"

    train_dir = "cars_train"
    test_dir = "cars_test"

    relevant_files = [train_dir, test_dir]

    def __init__(self, root, train: bool, transform=None, target_transform=None, download: bool = False,
                 cleanup: bool = True):
        self.train = train  # training set or test set
        self.root = root
        if download:
            self.download(cleanup)

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        data_root = os.path.join(root, self.train_dir if train else self.test_dir)
        super(Cars, self).__init__(data_root, transform=transform, target_transform=target_transform)

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

        download_and_extract_archive(self.ims_url, os.path.join(self.root, "raw_data"), filename=self.raw_ims_file)
        download_url(self.annos_url, os.path.join(self.root, "raw_data"), self.raw_annos_file)

        raw_annos = scipy.io.loadmat(os.path.join(self.root, "raw_data", self.raw_annos_file))

        # remove any "/" from file to not create weird subdirectories
        class_names = [x[0].replace("/", "") for x in raw_annos['class_names'][0]]
        for out_data_dir in (self.train_dir, self.test_dir):
            for cls in class_names:
                cls_path = os.path.join(self.root, out_data_dir, cls)
                if not os.path.exists(cls_path):
                    os.makedirs(cls_path)

        margin = 16
        for row in tqdm.tqdm(raw_annos['annotations'][0]):
            raw_filename = row['relative_im_path'][0]
            img_class = class_names[row['class'][0, 0] - 1]

            out_data_root_dir = os.path.join(self.root, self.test_dir if row['test'][0][0] else self.train_dir)
            dst_folder = os.path.join(out_data_root_dir, img_class)

            x1 = row['bbox_x1'][0, 0]
            x2 = row['bbox_x2'][0, 0]
            y1 = row['bbox_y1'][0, 0]
            y2 = row['bbox_y2'][0, 0]
            img_file = os.path.join(self.root, "raw_data", raw_filename)
            src_img = Image.open(img_file).convert('RGB')
            width, height = src_img.size
            # saves cropped image to file
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(x2 + margin, width - 1)
            y2 = min(y2 + margin, height - 1)

            dst_path = os.path.join(dst_folder, os.path.basename(raw_filename))
            crop_image = src_img.crop((x1, y1, x2, y2))

            with open(dst_path, "w") as f:
                crop_image.save(f)

        if cleanup:
            shutil.rmtree(os.path.join(self.root, "raw_data"))
