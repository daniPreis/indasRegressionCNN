import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from skimage import io
from torch.utils.data import Dataset


class ImageAnglesDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.angles_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.angles_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.angles_frame.iloc[idx, 0]

        image = io.imread(self.root_dir + '/' + str(img_name) + '.jpg')
        angles = np.array([self.angles_frame.iloc[idx, 2]]).astype(float)
        angles = angles / 360.0
        image = image.reshape(3, 400, 400)
        sample = {'image': image, 'Angle': angles}
        return sample


def show_angles(image):
    plt.imshow(image)
