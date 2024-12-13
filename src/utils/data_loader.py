import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

class EMNISTDatasetCSV(Dataset):
    __slots__ = ['data', 'transform', 'cache_dir', 'labels', 'images', 'label_to_char']

    def __init__(self, csv_file, mapping_file=None, transform=None, cache_dir=None):
        '''
        Initialize the EMNISTDatasetCSV class.
        Args:
            csv_file (str): Path to the EMNIST CSV file.
            mapping_file (str, optional): Path to the mapping file for label-to-character mapping.
            transform (callable, optional): Optional transform to be applied on an image.
        '''
        super().__init__()
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform
        self.cache_dir = cache_dir

        # split labels and images
        self.labels = self.data.iloc[:, 0].values  # first column is labels
        self.images = self.data.iloc[:, 1:].values  # rest are images

        # load mapping if provided
        self.label_to_char = {}
        if mapping_file:
            self.label_to_char = self._load_mapping(mapping_file)
            
        # create cache dir if specified
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
    def _preprocess_image(self, idx):
        image = self.images[idx].reshape(28, 28).astype(np.uint8)
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f'img_{idx}.npy')
            if os.path.exists(cache_path):
                return np.load(cache_path)      
            np.save(cache_path, image)
        return image
            
            

    def _load_mapping(self, mapping_file):
        '''Loads the label-to-character mapping from a text file.'''
        mapping = {}
        with open(mapping_file, 'r') as f:
            for line in f:
                label, char_code = line.strip().split()
                mapping[int(label)] = chr(int(char_code))
        return mapping

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # get image and label
        image = self.images[idx].reshape(28, 28).astype(np.uint8)
        label = self.labels[idx]

        # apply transform if provided
        if self.transform:
            image = self.transform(image)

        # return image and label
        if self.label_to_char:
            label = self.label_to_char[label]
        return image, label
