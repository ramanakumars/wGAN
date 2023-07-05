import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import pandas as pd
import numpy as np
from torchvision.transforms import Resize


class ImageDataset(Dataset):
    resize = Resize((256, 256))

    def __init__(self, imgfolder, label_file):
        self.imgfolder = imgfolder
        self.labels = pd.read_csv(label_file)

        self.subject_ids = np.asarray(self.labels.subject_id)

        print(f"Found data with {len(self)} images")

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, index):
        subject_id = self.subject_ids[index]
        fname = os.path.join(self.imgfolder, f'{subject_id}.png')
        img = self.resize(read_image(fname, ImageReadMode.RGB)) / 255.

        return img
