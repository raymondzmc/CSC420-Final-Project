import torch, os, pdb
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from scipy.misc import imread
import torch.nn.functional as F
import cv2 as cv

class FashionDataset(Dataset):
    '''
    Dataset for loading image and label for training
    '''
    def __init__(self, data_path, label_path, transform, mode='person', train=True, size=(224, 224)):

    	# Use the first 400 images for training and the others for testing
        self.img_list = [f for f in os.listdir(data_path) if f.split('.')[-1] == 'jpg']
        self.img_list.sort(key=lambda x:int(x.split('.')[0]))

        if train:
        	self.img_list = self.img_list[:400]
        else:
        	self.img_list = self.img_list[400:]

        self.train = train
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.mode = mode
        self.size = size

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        Training:
            Return a (C x 224 x 224) Tensor, where the first dimension
            represents a binary mask for each of the C classes
        Testing:
            Return (H x W) Tensor during testing, with each value
            corresponds to the index of the class it belongs to.
        """

        img_name = self.img_list[idx]

        # Load image
        img_path = os.path.join(self.data_path, img_name)
        img = Image.open(img_path)

        # Load training label
        label_name = '{}_{}.png'.format(img_name.split('.')[0], self.mode)
        label = imread(os.path.join(self.label_path, label_name), mode='P')

        # Horizontal flip the image with probability of 0.5
        if self.train and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = np.flip(label, axis=1)

        # Convert label to Tensors of required dimensions
        if self.mode == 'clothes':
            if self.train:
                label = np.tile(label, (7, 1, 1))
                multi_label = torch.zeros((7, *self.size))
                for i in range(7):

                    # Create binary mask for each class
                    true_mask = (label[i] == i)
                    false_mask = (label[i] != i)
                    label[i][true_mask] = 1
                    label[i][false_mask] = 0
                    multi_label[i] = torch.from_numpy(cv.resize(label[i], self.size))
                label = multi_label
            else:
                label = torch.from_numpy(label.copy())

        elif self.mode == 'person':
            if self.train:
                label = cv.resize(label, self.size)
            label = torch.from_numpy(label.copy()).unsqueeze(0)

        return self.transform(img), label.float(), img_name.split('.')[0]