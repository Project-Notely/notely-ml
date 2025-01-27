import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class IAMDataset(Dataset):
    def __init__(self, root_dir, annotations, transform=None):
        """
        Args:
            root_dir: Directory containing the images.
            annotations: CSV file with columns [image_path, transcription, writer_id].
            transform: Transformations for the images.
        """
        self.root_dir = root_dir
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path, transcription, writer_id = self.annotations[idx]
        image = Image.open(os.path.join(self.root_dir, img_path)).convert('L')  # grayscale
        if self.transform:
            image = self.transform(image)
        return image, transcription, writer_id
