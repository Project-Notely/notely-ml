import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image


class EMNISTPreprocessor:
    def __init__(self, root_dir, batch_size):
        
        # define default transformations
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # converts PIL Image or numpy.ndarray to tensor
                transforms.Normalize(
                    (0.5,), (0.5,)
                ),  # normalize pixel values to [-1, 1]
            ]
        )
        
        self.train_dataset = torchvision.datasets.EMNIST(
            root=root_dir,
            split="byclass",
            train=True,
            download=True,
            transform=self.transform,
        )

        self.test_dataset = torchvision.datasets.EMNIST(
            root=root_dir,
            split="byclass",
            train=False,
            download=True,
            transform=self.transform,
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )

        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )

    def get_transform(self):
        """Return the transformation pipeline"""
        return self.transform

    def set_custom_transform(self, transform):
        """Set a custom transformation pipeline"""
        self.transform = transform


class ImagePreprocessor:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def load_image(self, image_path):
        """Load an image from a file"""
        image = Image.open(image_path)
        image = image.resize((28, 28))
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor
