import torch
from torchvision import transforms
from PIL import Image

def preprocess_image(image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image)

def combine_images(images: list[torch.Tensor]) -> torch.Tensor:
    num_images = len(images)
    result = Image.new('L', (128 * num_images, 32))
    for i, img_tensor in enumerate(images):
        img = transforms.ToPILImage()(img_tensor)
        result.paste(img, (128 * i, 0))
    return result
