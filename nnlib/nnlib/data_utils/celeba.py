from torchvision import transforms, datasets
import torch

from .base import log_call_parameters
from .abstract import StandardVisionDataset


attribute_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
                   'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                   'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                   'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                   'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
                   'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                   'Wearing_Necktie', 'Young']


class CelebA(StandardVisionDataset):
    @log_call_parameters
    def __init__(self, data_augmentation: bool = False, target_type: str = 'attr', **kwargs):
        super(CelebA, self).__init__(**kwargs)
        self.data_augmentation = data_augmentation
        self.target_type = target_type

    @property
    def dataset_name(self) -> str:
        return 'celeba'

    @property
    def means(self):
        return torch.tensor([0.485, 0.456, 0.406])

    @property
    def stds(self):
        return torch.tensor([0.229, 0.224, 0.225])

    @property
    def train_transforms(self):
        if not self.data_augmentation:
            return self.test_transforms
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.CenterCrop(178),
                                   transforms.Resize(224),
                                   transforms.ToTensor(),
                                   self.normalize_transform])

    @property
    def test_transforms(self):
        return transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(224),
            transforms.ToTensor(),
            self.normalize_transform
        ])

    def raw_dataset(self, data_dir: str, download: bool, split: str, transform):
        assert split in ['train', 'val', 'test']
        if split == 'val':
            split = 'valid'
        return datasets.CelebA(data_dir, download=download, split=split, transform=transform,
                               target_type=self.target_type)
