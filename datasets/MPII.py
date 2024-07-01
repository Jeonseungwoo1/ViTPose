import os
import sys
import json
import random

import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets.transform import fliplr_joints, affine_transform, get_affine_transform

def calculate_bbox(joints):
        x_min = np.min(joints[:, 0])
        y_min = np.min(joints[:, 1])
        x_max = np.max(joints[:, 0])
        y_max = np.max(joints[:, 1])
        return [x_min, y_min, x_max - x_min, y_max - y_min]

class MPIIDataset(Dataset):
    """
    MPIIDataset class.
    """

    def __init__(self, root_path="/Users/jeonseung-u/Desktop/DeepLearning/ViTPose/datasets/mpii", data_version="train", 
                 is_train=True, image_width=288, image_height=384,
                 scale=True, scale_factor=0.35, flip_prob=0.5, rotate_prob=0.5, rotation_factor=45., half_body_prob=0.3,
                 use_different_joints_weight=False, heatmap_sigma=3):
        """
        Initializes a new MPIIDataset object.

        Image and annotation indexes are loaded and stored in memory.
        Annotations are preprocessed to have a simple list of annotations to iterate over.

        Args:
            root_path (str): dataset root path.
                Default: "./datasets/MPII"
            data_version (str): desired version/folder of MPII. Possible options are "train", "val".
                Default: "train"
            is_train (bool): train or eval mode. If true, train mode is used.
                Default: True
            image_width (int): image width.
                Default: 288
            image_height (int): image height.
                Default: 384
            scale (bool): scale mode.
                Default: True
            scale_factor (float): scale factor.
                Default: 0.35
            flip_prob (float): flip probability.
                Default: 0.5
            rotate_prob (float): rotate probability.
                Default: 0.5
            rotation_factor (float): rotation factor.
                Default: 45.
            half_body_prob (float): half body probability.
                Default: 0.3
            use_different_joints_weight (bool): use different joints weights.
                Default: False
            heatmap_sigma (float): sigma of the gaussian used to create the heatmap.
                Default: 3
        """
        super(MPIIDataset, self).__init__()

        self.root_path = root_path
        self.data_version = data_version
        self.is_train = is_train
        self.scale = scale
        self.scale_factor = scale_factor
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotation_factor = rotation_factor
        self.half_body_prob = half_body_prob
        self.use_different_joints_weight = use_different_joints_weight
        self.heatmap_sigma = heatmap_sigma

        # Image & annotation path
        self.data_path = f"{root_path}/images"
        self.annotation_path = f"/Users/jeonseung-u/Desktop/DeepLearning/ViTPose/datasets/mpii/annotation/train.json"

        self.image_size = (image_width, image_height)
        self.aspect_ratio = image_width * 1.0 / image_height

        self.heatmap_size = (int(image_width / 4), int(image_height / 4))
        self.heatmap_type = 'gaussian'
        self.pixel_std = 200

        self.num_joints = 16
        self.num_joints_half_body = 8

        # Define joint flip pairs for MPII dataset
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.upper_body_ids = list(range(8))
        self.lower_body_ids = list(range(8, 16))
        self.joints_weight = np.ones((self.num_joints, 1), dtype=np.float32)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load MPII dataset - Load images and annotations
        with open(self.annotation_path, 'r') as f:
            self.annotations = json.load(f)

        self.data = []
        # load annotations for each image of MPII
        for ann in tqdm(self.annotations, desc="Preparing images and annotations"):
            img_path = os.path.join(self.root_path, ann['image'])
            joints = np.array(ann['joints']).astype(np.float32)
            joints_visibility = np.array(ann['joints_vis']).astype(np.float32).reshape(-1, 1)
            joints_visibility = np.hstack((joints_visibility, joints_visibility))

            if 'bbox' not in ann:
                ann['bbox'] = calculate_bbox(joints)

            x, y, w, h = ann['bbox']
            center, scale = self._box2cs([x, y, w, h])

            self.data.append({
                'imgPath': img_path,
                'center': center,
                'scale': scale,
                'joints': joints,
                'joints_visibility': joints_visibility,
            })

        print('\nMPII dataset loaded!')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        joints_data = self.data[index].copy()

        # Load image
        try:
            image = np.array(Image.open(joints_data['imgPath']))
            if image.ndim == 2:
                # Some images are grayscale and will fail the transform, convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        except:
            raise ValueError(f"Fail to read {joints_data['imgPath']}")

        joints = joints_data['joints']
        joints_vis = joints_data['joints_visibility']

        c = joints_data['center']
        s = joints_data['scale']
        r = 0

        # Apply data augmentation
        if self.is_train:
            if (self.half_body_prob and random.random() < self.half_body_prob and
                np.sum(joints_vis[:, 0]) > self.num_joints_half_body):
                c_half_body, s_half_body = self._half_body_transform(joints, joints_vis)

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor

            if self.scale:
                s = s * np.clip(random.random() * sf + 1, 1 - sf, 1 + sf)

            if self.rotate_prob and random.random() < self.rotate_prob:
                r = np.clip(random.random() * rf, -rf * 2, rf * 2)
            else:
                r = 0

            if self.flip_prob and random.random() < self.flip_prob:
                image = image[:, ::-1, :]
                joints, joints_vis = fliplr_joints(joints, joints_vis, image.shape[1], self.flip_pairs)
                c[0] = image.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, self.pixel_std, r, self.image_size)
        image = cv2.warpAffine(
            image,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR
        )

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        if self.transform is not None:
            image = self.transform(image)

        target, target_weight = self._generate_target(joints, joints_vis)

        joints_data['joints'] = joints
        joints_data['joints_visibility'] = joints_vis
        joints_data['center'] = c
        joints_data['scale'] = s
        joints_data['rotation'] = r

        return image, target.astype(np.float32), target_weight.astype(np.float32), joints_data

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2,), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std], dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def _half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if random.random() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std], dtype=np.float32)
        scale = scale * 1.5

        return center, scale

    def _generate_target(self, joints, joints_vis):
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        if self.heatmap_type == 'gaussian':
            target = np.zeros((self.num_joints, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)

            tmp_size = self.heatmap_sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = np.asarray(self.image_size) / np.asarray(self.heatmap_size)
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0
                    continue

                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.heatmap_sigma ** 2))

                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        else:
            raise NotImplementedError

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight


if __name__ == '__main__':
    mpii = MPIIDataset(root_path=f"{os.path.dirname(__file__)}/mpii/images", data_version="train", rotate_prob=0., half_body_prob=0.)
    item = mpii[1]
    print(item[1].shape)
    print('ok!!')
    pass
