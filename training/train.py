import torch.optim as optim
from models.style_encoder import StyleEncoder
from models.content_encoder import ContentEncoder
from models.generator import Generator
from models.discriminator import Discriminator
from utils.data_loader import DataLoader
import torch

# Initialize models and optimizers
style_encoder = StyleEncoder()
content_encoder = ContentEncoder(vocab_size=100)  # Adjust vocab size
generator = Generator(style_dim=128, content_dim=256)
discriminator = Discriminator()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

# Example training loop
for epoch in range(num_epochs):
    for images, transcriptions, writer_ids in dataloader:
        # 1. Encode style and content
        style_features = style_encoder(images)
        content_features = content_encoder(transcriptions)

        # 2. Generate fake images
        fake_images = generator(style_features, content_features)

        # 3. Train Discriminator
        real_validity = discriminator(images)
        fake_validity = discriminator(fake_images.detach())
        d_loss = -torch.mean(torch.log(real_validity) + torch.log(1 - fake_validity))
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # 4. Train Generator
        fake_validity = discriminator(fake_images)
        g_loss = -torch.mean(torch.log(fake_validity))
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
