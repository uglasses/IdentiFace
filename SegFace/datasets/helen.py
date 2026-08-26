import os
import numpy as np
import cv2
import re
import torch
import torchvision
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from skimage import io
from PIL import Image, ImageDraw
import random
import warnings
import math
warnings.filterwarnings("ignore")


def get_image_name(original_name):
    final_name = f"{original_name}_image.jpg"
    return final_name

def get_label_name(original_name):
    final_name = f"{original_name}_label.png"
    return final_name


class HELEN(Dataset):
    """
    annotations
        bg
        face
        lb
        rb
        le
        re
        nose
        ulip
        imouth
        llip
        hair
    """
    def __init__(self, root, split, resolution):
        assert os.path.isdir(root)
        self.root=root
        self.split = split
        self.resolution = resolution
        if self.split=='train':
            self.filenames = os.listdir(os.path.join(self.root, 'train'))
            unique_filenames = set()
            pattern = re.compile(r'(_image|_label|_viz)')
            for filename in self.filenames:
                base_name = pattern.sub('', filename).rsplit('.', 1)[0]
                unique_filenames.add(base_name)
            self.filenames = sorted(list(unique_filenames))
        else:
            self.filenames = os.listdir(os.path.join(self.root, 'test'))
            unique_filenames = set()
            pattern = re.compile(r'(_image|_label|_viz)')
            for filename in self.filenames:
                base_name = pattern.sub('', filename).rsplit('.', 1)[0]
                unique_filenames.add(base_name)
            self.filenames = sorted(list(unique_filenames))
        
        self.file_path = os.path.join(self.root, 'landmarks.txt')
        self.transforms_image = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transforms_image_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
 
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        self.landmarks = {}
        for line in lines:
            line = line.strip().split()
            filename = str(line[-1])
            landmark = np.asarray(line[0:136], dtype=np.float32).reshape(68,2)
            self.landmarks[filename] = landmark


    def __len__(self):
        return len(self.filenames)

    def label_names(self):
        return ["bg", "face", "lb", "rb", "le", "re", "nose", "ulip", "imouth", "llip", "hair"]

    def __getitem__(self, index):
        if self.split=="train":
            image_path = os.path.join(self.root, self.split, get_image_name(self.filenames[index]))
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = cv2.imread(os.path.join(self.root, self.split, get_label_name(self.filenames[index])), cv2.IMREAD_GRAYSCALE)
            landmark = self.landmarks[self.filenames[index]]
            landmark = landmark * np.array([1/2.0, 1/2.0])

            ## Bounding Box from Landmarks ##
            image = np.array(image)
            label = np.array(label)
            xy = np.min(landmark, axis=0).astype(np.float32)
            zz = np.max(landmark, axis=0).astype(np.float32)
            wh = zz - xy + 1
            center = (xy + wh/2).astype(np.int32)
            boxsize = int(np.max(wh)*1.2)
            xy = center - boxsize//2
            x1, y1 = xy
            x2, y2 = xy + boxsize
            height, width, _ = image.shape
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)
            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)
            image = image[y1:y2, x1:x2]
            label = label[y1:y2, x1:x2]

            # Update landmarks after cropping
            landmark[:, 0] -= x1
            landmark[:, 1] -= y1

            h, w = image.shape[:2]
            image = Image.fromarray(image)
            label = Image.fromarray(label)

            # Resize image and label to the desired size
            image = F.resize(image, size=(self.resolution, self.resolution), interpolation=Image.BICUBIC)
            label = F.resize(label, size=(self.resolution, self.resolution), interpolation=Image.NEAREST)

            original_size = np.array([w, h])
            new_size = np.array([self.resolution, self.resolution])
            scale_x = new_size[0] / original_size[0]
            scale_y = new_size[1] / original_size[1]

            # Apply resizing
            landmark = landmark * np.array([scale_x, scale_y])

            # Convert to tensor
            image = self.transforms_image(image)
            label = F.to_tensor(label)
            label = torch.squeeze(label) * 255  # Assuming label images are in grayscale
            label = label.to(dtype=torch.float)

            landmarks_five = []
            landmarks_five.append([(landmark[36][0] + landmark[37][0] + landmark[38][0] + landmark[39][0] + landmark[40][0] + landmark[41][0])/6, (landmark[36][1] + landmark[37][1] + landmark[38][1] + landmark[39][1] + landmark[40][1] + landmark[41][1])/6])
            landmarks_five.append([(landmark[42][0] + landmark[43][0] + landmark[44][0] + landmark[45][0] + landmark[46][0] + landmark[47][0])/6, (landmark[42][1] + landmark[43][1] + landmark[44][1] + landmark[45][1] + landmark[46][1] + landmark[47][1])/6])
            landmarks_five.append([landmark[30][0], landmark[30][1]])
            landmarks_five.append([landmark[48][0], landmark[48][1]])
            landmarks_five.append([landmark[54][0], landmark[54][1]])
            landmarks_five = np.array(landmarks_five)
            landmarks_five = torch.tensor(landmarks_five, dtype=torch.float)

            data = {'image': image, 'label': {"segmentation":label, "lnm_seg": landmarks_five}, "dataset": 2}
            return data
        else:
            image_path = os.path.join(self.root, self.split, get_image_name(self.filenames[index]))
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = cv2.imread(os.path.join(self.root, self.split, get_label_name(self.filenames[index])), cv2.IMREAD_GRAYSCALE)
            landmark = self.landmarks[self.filenames[index]]
            landmark = landmark * np.array([1/2.0, 1/2.0])

            ## Bounding Box from Landmarks ##
            image = np.array(image)
            label = np.array(label)
            xy = np.min(landmark, axis=0).astype(np.float32)
            zz = np.max(landmark, axis=0).astype(np.float32)
            wh = zz - xy + 1
            center = (xy + wh/2).astype(np.int32)
            boxsize = int(np.max(wh)*1.2)
            xy = center - boxsize//2
            x1, y1 = xy
            x2, y2 = xy + boxsize
            height, width, _ = image.shape
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)
            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)
            image = image[y1:y2, x1:x2]
            label = label[y1:y2, x1:x2]

            # Update landmarks after cropping
            landmark[:, 0] -= x1
            landmark[:, 1] -= y1

            h, w = image.shape[:2]
            image = Image.fromarray(image)
            label = Image.fromarray(label)

            # Resize image and label to the desired size
            image = F.resize(image, size=(self.resolution, self.resolution), interpolation=Image.BICUBIC)
            label = F.resize(label, size=(self.resolution, self.resolution), interpolation=Image.NEAREST)

            original_size = np.array([w, h])
            new_size = np.array([self.resolution, self.resolution])
            scale_x = new_size[0] / original_size[0]
            scale_y = new_size[1] / original_size[1]

            # Apply resizing
            landmark = landmark * np.array([scale_x, scale_y])


            # Convert to tensor
            image = self.transforms_image_test(image)
            label = F.to_tensor(label)
            label = torch.squeeze(label) * 255  # Assuming label images are in grayscale
            label = label.to(dtype=torch.float)

            landmarks_five = []
            landmarks_five.append([(landmark[36][0] + landmark[37][0] + landmark[38][0] + landmark[39][0] + landmark[40][0] + landmark[41][0])/6, (landmark[36][1] + landmark[37][1] + landmark[38][1] + landmark[39][1] + landmark[40][1] + landmark[41][1])/6])
            landmarks_five.append([(landmark[42][0] + landmark[43][0] + landmark[44][0] + landmark[45][0] + landmark[46][0] + landmark[47][0])/6, (landmark[42][1] + landmark[43][1] + landmark[44][1] + landmark[45][1] + landmark[46][1] + landmark[47][1])/6])
            landmarks_five.append([landmark[30][0], landmark[30][1]])
            landmarks_five.append([landmark[48][0], landmark[48][1]])
            landmarks_five.append([landmark[54][0], landmark[54][1]])
            landmarks_five = np.array(landmarks_five)
            landmarks_five = torch.tensor(landmarks_five, dtype=torch.float)

            data = {'image': image, 'label': {"segmentation":label, "lnm_seg": landmarks_five}, "dataset": 2}
            return data


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

    @staticmethod
    def draw_landmarks(im, landmarks, color, thickness=3, eye_radius=0):
        im = np.ascontiguousarray(im)
        landmarks = landmarks.numpy().astype(np.int32)
        for (x, y) in landmarks:
            cv2.circle(im, (x,y), eye_radius, color, thickness)
        return im

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

    @property
    def label_names(self):
        return ['bg', 'face', 'lb', 'rb', 'le', 're', 'nose', 'ulip', 'imouth', 'llip', 'hair']
        
def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    tensor = tensor * std + mean  # Apply the unnormalize formula
    tensor = torch.clamp(tensor, 0, 1)  # Clamp the values to be between 0 and 1
    return tensor



if __name__ == '__main__':
    helen_parse = HELEN("/data/knaraya4/data/SegFace/helen", 'train', resolution=512)
    print("Length of Helen parse test set: ", len(helen_parse))
    loader = torch.utils.data.DataLoader(helen_parse, batch_size=1, shuffle=False, num_workers=4)

    # Check batch
    for i, batch in enumerate(loader):
        # Save face image
        print(f"Batch_{i}")
        face = unnormalize(batch['image'][0]).permute(1, 2, 0).numpy()
        face = (face * 255).astype(np.uint8)
        cv2.imwrite(f"/data/knaraya4/SegFace/samples/face_new_{i}.png", face[:, :, ::-1])

        # Save visualized mask
        mask = helen_parse.visualize_mask(batch["label"]['segmentation'][0].numpy())
        cv2.imwrite(f"/data/knaraya4/SegFace/samples/mask_new_{i}.png", mask[:, :, ::-1])

        if i >= 19:
            break