from PIL import Image

from torch.utils.data import Dataset
import tensorflow_datasets as tfds


class SimpleTensorFlowDataset(Dataset):
    def __init__(self, transform, **kwargs):
        super(SimpleTensorFlowDataset, self).__init__()

        self.transform = transform
        tf_dataset = tfds.load(**kwargs)
        if isinstance(tf_dataset, dict):
            raise ValueError("More than one split is returned")

        self._dataset = []
        for image, label in tf_dataset:
            image = Image.fromarray(image.numpy())
            self._dataset.append((image, int(label)))

    def __getitem__(self, index):
        img, label = self._dataset[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self._dataset)
