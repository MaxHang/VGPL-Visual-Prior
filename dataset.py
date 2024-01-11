import os

import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader


# paths to data directories
_DATA_DIR = {
    'RigidFall': '/datasets/VGPL-Visual-Prior/datasets/data_RigidFall/',
    'MassRope': '/datasets/VGPL-Visual-Prior/datasets/data_MassRope/',
    'CConvFluid': '/datasets/VGPL-Visual-Prior/datasets/data_cconv_fluid_6kbox_21times_120X160',
    'CConvFluid801times': '/datasets/VGPL-Visual-Prior/datasets/data_cconv_fluid_6kbox_801times_120X160',
    # 'RigidFall': './data/data_RigidFall/',
    # 'MassRope': './data/data_MassRope/',
}

# number of frames dropped from beginning of each video
_N_DROP_FRAMES = {
    'RigidFall': 0,
    'MassRope': 20,
    'CConvFluid': 0,
    'CConvFluid801times': 0,
}

# image mean and std for standardization of all datasets
_IMG_MEAN = 0.575
_IMG_STD = 0.375

# 定义绕X轴旋转90°的旋转矩阵
rotation_matrix_x = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
], dtype=np.float32)

# 定义绕Z轴旋转180°的旋转矩阵
rotation_matrix_z = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
], dtype=np.float32)
# 合并两个旋转矩阵
# combined_rotation_matrix = np.dot(rotation_matrix_z, rotation_matrix_x)



def _load_data(path):
    """Load h5py data files from specified path."""
    hf = h5py.File(path, 'r')
    data = []
    for dn in ['images', 'positions']:
        d = np.array(hf.get(dn))
        data.append(d)
    hf.close()
    return data


def normalize(images):
    """
    Normalize an input image array.

    Input:
        images: numpy array of shape (T, H, W, C)

    Returns:
        images: numpy array of shape (T, C, H, W)
    """
    images = (images - _IMG_MEAN) / _IMG_STD

    # (T, C, H, W)
    images = np.transpose(images, (0, 3, 1, 2))

    return images


def denormalize(images):
    """
    De-normalize an input image array.

    Input:
        images: numpy array of shape (T, H, W, C)

    Output:
        images: numpy array of shape (T, C, H, W)
    """
    images = images * _IMG_STD + _IMG_MEAN

    # (T, H, W, C)
    images = np.transpose(images, (0, 2, 3, 1))

    return images


class PhyDataset(Dataset):
    """
    Dataset for physical scene observations.
    Available environments: 'RigidFall', 'MassRope'
    """

    def __init__(self, name, split):
        if name not in ['RigidFall', 'MassRope', 'CConvFluid', 'CConvFluid801times']:
            raise ValueError('Invalid dataset name {}.'.format(name))
        if split not in ['train', 'valid']:
            raise ValueError('Invalid dataset split {}.'.format(split))

        self._name = name
        self._split = split
        self._data_dir = os.path.join(
            _DATA_DIR[name], '{}_vision'.format(split))
        self._data_len = None

    def __len__(self):
        if self._data_len is None:
            self._data_len = len(os.listdir(self._data_dir))
        return self._data_len

    def __getitem__(self, idx):
        if idx >= len(self) or idx < 0:
            raise ValueError('Invalid index {}'.format(idx))

        data_path = os.path.join(self._data_dir, str(idx) + '.h5')
        images, positions = _load_data(data_path)

        # Remove environment particles
        if self._name == 'RigidFall':
            positions = positions[:, :-1, :]
            groups = np.array([0 for _ in range(64)]
                              + [1 for _ in range(64)]
                              + [2 for _ in range(64)])
        elif self._name == 'MassRope':  # 'MassRope'
            positions = positions[:-1, :-1, :]
            groups = np.array([0 for _ in range(81)]
                              + [1 for _ in range(14)])
        elif self._name == 'CConvFluid':
            groups = np.array([0 for _ in range(6000)])
            # 右手坐标系, 将顶点绕x旋转90, 绕z旋转180 即 x,y,z -> x,-z,y ->-x,z,y
            positions[..., 0] *= -1
            positions[..., [1,2]] = positions[..., [2,1]]

        elif self._name == 'CConvFluid801times':
            groups = np.array([0 for _ in range(6000)])
            # positions = np.dot(positions, combined_rotation_matrix.T)
            positions[..., 0] *= -1
            positions[..., [1,2]] = positions[..., [2,1]]
        else:
            pass

        # Drop frames from beginning of video
        n_drop = _N_DROP_FRAMES[self._name]
        images = images[n_drop:, ...]
        positions = positions[n_drop:, ...]

        images = images.astype(np.float32) / 255
        images = normalize(images)

        return images, positions, groups


def get_dataloader(config, split, shuffle=None):
    name = config.dataset
    batch_size = config.batch_size
    if shuffle is None:
        shuffle = (split == 'train')

    ds = PhyDataset(name, split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
