import os
import numpy as np
import cv2
import functools
import torch
import torchvision
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms.functional as F
from PIL import Image
import random
from typing import List
import warnings
warnings.filterwarnings("ignore")

@functools.lru_cache()
def _cached_imread(fname, flags=None):
    return cv2.imread(fname, flags=flags)

class CelebAMaskHQ(Dataset):
    def __init__(self, root, split, resolution,label_type='all'):
        assert os.path.isdir(root)
        self.resolution = resolution
        self.root = root
        self.split = split
        self.names = []

        if split != 'all':
            hq_to_orig_mapping = dict()
            orig_to_hq_mapping = dict()
            mapping_file = os.path.join(
                root, 'CelebA-HQ-to-CelebA-mapping.txt')
            assert os.path.exists(mapping_file)
            for s in open(mapping_file, 'r'):
                if '.jpg' not in s:
                    continue
                idx, _, orig_file = s.split()
                hq_to_orig_mapping[int(idx)] = orig_file
                orig_to_hq_mapping[orig_file] = int(idx)

            # load partition
            partition_file = os.path.join(root, 'list_eval_partition.txt')
            assert os.path.exists(partition_file)
            for s in open(partition_file, 'r'):
                if '.jpg' not in s:
                    continue
                orig_file, group = s.split()
                group = int(group)
                if orig_file not in orig_to_hq_mapping:
                    continue
                hq_id = orig_to_hq_mapping[orig_file]
                if split == 'train' and group == 0:
                    self.names.append(str(hq_id))
                elif split == 'val' and group == 1:
                    self.names.append(str(hq_id))
                elif split == 'test' and group == 2:
                    self.names.append(str(hq_id))
        else:
            self.names = [
                n[:-(len('.jpg'))]
                for n in os.listdir(os.path.join(self.root, 'CelebA-HQ-img'))
                if n.endswith('.jpg')
            ]

        self.label_setting = {
            'human': {
                'suffix': [
                    'neck', 'skin', 'cloth', 'l_ear', 'r_ear', 'l_brow', 'r_brow',
                    'l_eye', 'r_eye', 'nose', 'mouth', 'l_lip', 'u_lip', 'hair'
                ],
                'names': [
                    'bg', 'neck', 'face', 'cloth', 'rr', 'lr', 'rb', 'lb', 're',
                    'le', 'nose', 'imouth', 'llip', 'ulip', 'hair'
                ]
            },
            'aux': {
                'suffix': [
                    'eye_g', 'hat', 'ear_r', 'neck_l',
                ],
                'names': [
                    'normal', 'glass', 'hat', 'earr', 'neckl'
                ]
            },
            'all': {
                'suffix': [
                    'neck', 'skin', 'cloth', 'l_ear', 'r_ear', 'l_brow', 'r_brow',
                    'l_eye', 'r_eye', 'nose', 'mouth', 'l_lip', 'u_lip', 'hair',
                    'eye_g', 'hat', 'ear_r', 'neck_l',
                ],
                'names': [
                    'bg', 'neck', 'face', 'cloth', 'lr', 'rr', 'lb', 'rb', 'le',
                    're', 'nose', 'imouth', 'llip', 'ulip', 'hair',
                    'glass', 'hat', 'earr', 'neckl'
                ]
            }
        }[label_type]

        self.transforms_image = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transforms_image_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def make_label(self, index, ordered_label_suffix):
        label = np.zeros((512, 512), np.uint8)
        name = self.names[index]
        name_id = int(name)
        name5 = '%05d' % name_id
        p = os.path.join(self.root, 'CelebAMask-HQ-mask-anno',
                         str(name_id // 2000), name5)
        for i, label_suffix in enumerate(ordered_label_suffix):
            label_value = i + 1
            label_fname = os.path.join(p + '_' + label_suffix + '.png')
            if os.path.exists(label_fname):
                mask = _cached_imread(label_fname, cv2.IMREAD_GRAYSCALE)
                label = np.where(mask > 0, np.ones_like(label) * label_value, label)
        return label

    def __getitem__(self, index):
        if self.split=='train':
            name = self.names[index]
            image = cv2.imread(os.path.join(self.root, 'CelebA-HQ-img',name + '.jpg'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = self.make_label(index, self.label_setting['suffix'])

            # Convert to PIL for compatibility with torchvision functional transforms
            image = Image.fromarray(image)
            label = Image.fromarray(label)

            # Resize image and label to the desired size
            image = F.resize(image, size=(self.resolution, self.resolution), interpolation=Image.BICUBIC)
            label = F.resize(label, size=(self.resolution, self.resolution), interpolation=Image.NEAREST)

            # Convert to tensor
            image = self.transforms_image(image)
            label = F.to_tensor(label)
            label = torch.squeeze(label) * 255  # Assuming label images are in grayscale
            label = label.to(dtype=torch.float)

            data = {'image': image, 'label': {"segmentation":label, "lnm_seg": torch.zeros([5,2])}, "dataset": 0}
            return data
        else:
            name = self.names[index]
            image = cv2.imread(os.path.join(self.root, 'CelebA-HQ-img',name + '.jpg'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = self.make_label(index, self.label_setting['suffix'])

            # Convert to PIL for compatibility with torchvision functional transforms
            image = Image.fromarray(image)
            label = Image.fromarray(label)

            # Resize image and label to the desired size
            image = F.resize(image, size=(self.resolution, self.resolution), interpolation=Image.BICUBIC)
            label = F.resize(label, size=(self.resolution, self.resolution), interpolation=Image.NEAREST)

            # Convert to tensor
            image = self.transforms_image_test(image)
            label = F.to_tensor(label)
            label = torch.squeeze(label) * 255  # Assuming label images are in grayscale
            label = label.to(dtype=torch.float)

            data = {'image': image, 'label': {"segmentation":label, "lnm_seg": torch.zeros([5,2])}, "dataset": 0}
            return data

    def __len__(self):
        return len(self.names)

    def sample_name(self, index):
        return self.names[index]

    @property
    def label_names(self) -> List[str]:
        return self.label_setting['names']

    @staticmethod
    def visualize_mask(mask):
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color_mapping = np.array([
            [0, 0, 0], 
            [204, 0, 0], 
            [76, 153, 0],
            [204, 204, 0], 
            [51, 51, 255], 
            [204, 0, 204], 
            [0, 255, 255],
            [51, 255, 255], 
            [102, 51, 0], 
            [255, 0, 0], 
            [102, 204, 0],
            [255, 255, 0], 
            [0, 0, 153], 
            [0, 0, 204], 
            [255, 51, 153], 
            [0, 204, 204], 
            [0, 51, 0], 
            [255, 153, 51], 
            [0, 204, 0]
        ], dtype=np.uint8)

        for index, color in enumerate(color_mapping):
            i, j = np.where(mask == index)
            color_mask[i, j] = color
        return color_mask

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    tensor = tensor * std + mean  # Apply the unnormalize formula
    tensor = torch.clamp(tensor, 0, 1)  # Clamp the values to be between 0 and 1
    return tensor


if __name__ == '__main__':
    celebamaskhq = CelebAMaskHQ("/data/knaraya4/data/SegFace/CelebAMask-HQ", 'train', 512)
    loader = torch.utils.data.DataLoader(celebamaskhq, batch_size=8, shuffle=True, num_workers=4)

    print(celebamaskhq.label_names)
    # # Check batch
    # for i, batch in enumerate(loader):
    #     # Save face image
    #     face = unnormalize(batch['image'][0]).permute(1, 2, 0).numpy()
    #     face = (face * 255).astype(np.uint8)
    #     cv2.imwrite(f"/data/knaraya4/SegFace/samples/face_{i}.png", face[:, :, ::-1])

    #     # Save visualized mask
    #     mask = celebamaskhq.visualize_mask(batch["label"]['segmentation'][0].numpy())
    #     cv2.imwrite(f"/data/knaraya4/SegFace/samples/mask_{i}.png", mask[:, :, ::-1])

    #     if i >= 19:
    #         break


    # celebamaskhq = CelebAMaskHQ("/data/knaraya4/data/SegFace/CelebAMask-HQ", 'train', 512)
    # loader = torch.utils.data.DataLoader(celebamaskhq, batch_size=8, shuffle=True, num_workers=4)

    # class_frequencies = np.zeros(len(celebamaskhq.label_names), dtype=int)

    # total_count = 0
    # for i, batch in enumerate(loader):
    #     labels = batch["label"]["segmentation"].numpy()
    #     for label in labels:
    #         unique = np.unique(label)
    #         for u in unique:
    #             class_frequencies[int(u)] += 1
    #         total_count += 1

    # # Print class frequencies
    # for class_name, frequency in zip(celebamaskhq.label_names, class_frequencies):
    #     print(f"{class_name}: {frequency}")

    # print("Total images: ", total_count)