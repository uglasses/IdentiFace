import os
import numpy as np
import cv2
import torch
import torchvision
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms.functional as F
from PIL import Image
import random
import warnings
import math
warnings.filterwarnings("ignore")


class LaPa(Dataset):
    """LaPa face parsing dataset

    Args:
        root (str): The directory that contains subdirs 'image', 'labels'
        split (str): 'train, 'test' or 'val' split
    """

    def __init__(self, root, split, resolution):
        super().__init__()
        assert os.path.isdir(root)
        self.resolution = resolution
        self.root = root
        self.split = split

        subfolders = []
        if split == 'train':
            subfolders = ['train']
        elif split == 'val':
            subfolders = ['val']
        elif split == 'test':
            subfolders = ['test']
        elif split == 'all':
            subfolders = ['train', 'val', 'test']

        self.info = []
        for subf in subfolders:
            for name in os.listdir(os.path.join(self.root, subf, 'images')):
                if not name.endswith('.jpg'):
                    continue
                name = name.split('.')[0]
                image_path = os.path.join(
                    self.root, subf, 'images', f'{name}.jpg')
                label_path = os.path.join(
                    self.root, subf, 'labels', f'{name}.png')
                landmark_path = os.path.join(
                    self.root, subf, 'landmarks', f'{name}.txt')
                assert os.path.exists(image_path)
                assert os.path.exists(label_path)
                assert os.path.exists(landmark_path)
                landmarks = [float(v) for v in open(
                    landmark_path, 'r').read().split()]
                assert landmarks[0] == 106 and len(landmarks) == 106*2+1
                landmarks = np.reshape(
                    np.array(landmarks[1:], np.float32), [106, 2])
                sample_name = f'{subf}.{name}'
                self.info.append(
                    {'image_path': image_path, 'label_path': label_path,
                     'landmarks': landmarks, 'sample_name': sample_name})

        self.transforms_image = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transforms_image_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        

    def __getitem__(self, index):
        if self.split == "train":
            info = self.info[index]
            image = cv2.imread(info['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = cv2.imread(info['label_path'], cv2.IMREAD_GRAYSCALE)
            landmark = info['landmarks']

            h, w = image.shape[:2]
            image = Image.fromarray(image)
            label = Image.fromarray(label)

            # # Randomly apply transformations with the same parameters to both image and label
            center = np.array([w, h]) / 2.0
            angle = random.uniform(-30, 30)  # Rotation angle
            scale = random.uniform(0.5, 3)  # Scaling factor
            max_shift = 20  # Maximum shift in pixels
            translate = (random.uniform(-max_shift, max_shift), random.uniform(-max_shift, max_shift))  # Translation
            
            image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=0, interpolation=Image.BICUBIC)
            label = F.affine(label, angle=angle, translate=translate, scale=scale, shear=0, interpolation=Image.NEAREST)
            landmarks_transformed = self.apply_affine_transform_to_landmarks(landmark, center, angle, scale, translate)

            # # Resize image and label to the desired size
            image = F.resize(image, size=(self.resolution, self.resolution), interpolation=Image.BICUBIC)
            label = F.resize(label, size=(self.resolution, self.resolution), interpolation=Image.NEAREST)


            original_size = np.array([w, h])
            new_size = np.array([self.resolution, self.resolution])
            scale_x = new_size[0] / original_size[0]
            scale_y = new_size[1] / original_size[1]
            
            # Apply resizing
            landmark = landmarks_transformed * np.array([scale_x, scale_y])

            # landmark = landmark * [self.resolution/w, self.resolution/h]
            image = self.transforms_image(image)
            label = F.to_tensor(label)
            label = torch.squeeze(label) * 255  # Assuming label images are in grayscale
            label = label.to(dtype=torch.float)  

            landmarks_five = []
            landmarks_five.append([(landmark[74][0] + landmark[104][0])/2, (landmark[74][1] + landmark[104][1])/2])
            landmarks_five.append([(landmark[83][0] + landmark[105][0])/2, (landmark[83][1] + landmark[105][1])/2])
            landmarks_five.append([landmark[54][0], landmark[54][1]])
            landmarks_five.append([landmark[84][0], landmark[84][1]])
            landmarks_five.append([landmark[90][0], landmark[90][1]])
            landmarks_five = np.array(landmarks_five)
            landmarks_five = torch.tensor(landmarks_five, dtype=torch.float)

            data = {'image': image, 'label': {"segmentation":label, "lnm_seg": landmarks_five}, "dataset": 1}
            return data
        else:
            info = self.info[index]
            image = cv2.imread(info['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = cv2.imread(info['label_path'], cv2.IMREAD_GRAYSCALE)
            landmark = info['landmarks']
            h, w = image.shape[:2]

            image = Image.fromarray(image)
            label = Image.fromarray(label)
            image = F.resize(image, size=(self.resolution, self.resolution), interpolation=Image.BICUBIC)
            label = F.resize(label, size=(self.resolution, self.resolution), interpolation=Image.NEAREST)
            image = self.transforms_image_test(image)
            label = F.to_tensor(label)
            label = torch.squeeze(label) * 255  # Assuming label images are in grayscale
            label = label.to(dtype=torch.float)

            # Landmarks for Tanh Warping
            landmarks_five = []
            landmarks_five.append([(landmark[74][0] + landmark[104][0])/2, (landmark[74][1] + landmark[104][1])/2])
            landmarks_five.append([(landmark[83][0] + landmark[105][0])/2, (landmark[83][1] + landmark[105][1])/2])
            landmarks_five.append([landmark[54][0], landmark[54][1]])
            landmarks_five.append([landmark[84][0], landmark[84][1]])
            landmarks_five.append([landmark[90][0], landmark[90][1]])
            landmarks_five = np.array(landmarks_five)
            scale_x = self.resolution / w
            scale_y = self.resolution / h
            landmarks_five[:,0] *= scale_x
            landmarks_five[:,1] *= scale_y
            landmarks_five = torch.tensor(landmarks_five, dtype=torch.float)


            data = {'image': image, 'label': {"segmentation":label, "lnm_seg": landmarks_five}, "dataset": 1}
            return data            

    def __len__(self):
        return len(self.info)

    def sample_name(self, index):
        return self.info[index]['sample_name']

    @property
    def label_names(self):
        return ['background', 'face_lr_rr', 'lb', 'rb', 'le', 're', 'nose', 'ul', 'im', 'll', 'hair']

    def apply_affine_transform_to_landmarks(self, landmarks, center, angle, scale, translate):
        # Convert angle to radians and create rotation matrix
        angle_rad = np.deg2rad(-angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])
        
        # Adjust landmarks to rotate around the center of the image
        landmarks_centered = landmarks - center
        
        # Apply rotation and scaling
        landmarks_transformed = np.dot(landmarks_centered, rotation_matrix) * scale
        
        # Re-center landmarks and apply translation
        landmarks_transformed += center + np.array(translate)
        
        return landmarks_transformed

    @staticmethod
    def draw_landmarks(im, landmarks, color, thickness=5, eye_radius=3):
        im = im.permute(1, 2, 0).numpy()
        im = (im * 255).astype(np.uint8)
        im = np.ascontiguousarray(im)
        landmarks = landmarks.numpy().astype(np.int32)
        for i in range(landmarks.shape[0]):
            cv2.circle(im, (landmarks[i, 0], landmarks[i, 1]), eye_radius, color, -1, cv2.LINE_AA)
        return im

    @staticmethod
    def visualize_mask(mask):
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color_mapping = np.array([
            [0, 0, 0],
            [0, 153, 255],
            [102, 255, 153],
            [0, 204, 153],
            [255, 255, 102],
            [255, 255, 204],
            [255, 153, 0],
            [255, 102, 255],
            [102, 0, 51],
            [255, 204, 255],
            [255, 0, 102]
        ])
        for index, color in enumerate(color_mapping):
            i, j = np.where(mask == index)
            color_mask[i, j] = color
        return color_mask

if __name__ == '__main__':
    lapa = LaPa("/data/knaraya4/data/SegFace/LaPa", 'test', resolution=512)
    loader = torch.utils.data.DataLoader(lapa, batch_size=8, shuffle=True, num_workers=4)

    # Check batch
    for i, batch in enumerate(loader):
        # Save face image
        face = batch['image'][0].permute(1, 2, 0).numpy()
        face = (face * 255).astype(np.uint8)
        cv2.imwrite(f"/data/knaraya4/SegFace/samples/face_{i}.png", face[:, :, ::-1])

        # Save visualized mask
        mask = lapa.visualize_mask(batch["label"]['segmentation'][0].numpy())
        cv2.imwrite(f"/data/knaraya4/SegFace/samples/mask_{i}.png", mask[:, :, ::-1])

        if i >= 19:
            break
