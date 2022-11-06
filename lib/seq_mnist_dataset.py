import os
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class SequentialMNIST(Dataset):
    """
    Sequential multi-mnist dataset
    """
    def __init__(self, root, mode, seq_len):
        """
        Args:
            root: directory path
            mode: either ['train', 'valid']
            len: sequence length
        """
        assert mode in ['train', 'valid'], 'Invalid dataset mode'
        path = {"train": "train", "valid": "test"}[mode]
        super().__init__()
        # Path to the pickle file
        path = os.path.join(root, path)

        self.sequences = list(sorted(Path(path).glob("*")))
        self.anns = [json.load(p.joinpath("annotations.json").open()) for p in self.sequences]

        # (T, N, H, W)

    @staticmethod
    def load_image(path):
        return np.array(Image.open(path))[:, :, [0]]


    def __getitem__(self, index):
        """
        Args:
            index: integer
        Returns:
            A tuple (imgs, nums).

            imgs: (T, 1, C, H, W), FloatTensor in range (0, 1)
            nums: (T, 1), int, number of digits for each time step
        """
        sequence = self.sequences[index]
        ann = self.anns[index]

        images_files = list(sorted(sequence.glob("*.jpg")))
        imgs = np.stack(self.load_image(p) for p in images_files).transpose(0, 3, 1, 2)
        nums = [len(a["bboxes"]) for a in ann]

        imgs = imgs.astype(np.float) / 255.0
        imgs = torch.from_numpy(imgs).float()
        nums = torch.Tensor(nums).float()

        imgs = imgs.unsqueeze(1)
        nums = nums.unsqueeze(-1)

        return imgs, nums

    def __len__(self):
        return len(self.sequences)


def collate_fn(samples):
    """
    collate_fn for SequentialMNIST.

    Args:
        samples: a list of samples. Each item is a (imgs, nums) pair. Where

            - imgs: shape (T, 1, C, H, W)
            - nums: shape (T, 1)

            And len(samples) is the batch size.

    Returns:
        A tuple (imgs, nums). Where

        - imgs: shape (T, B, C, H, W)
        - nums: shape (T, B)
    """
    imgs, nums = zip(*samples)
    # (T, B, C, H, W)
    imgs = torch.cat(imgs, dim=1)
    # (T, B)
    nums = torch.cat(nums, dim=1)

    return imgs, nums



