import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
from scipy.stats import entropy
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torchvision.transforms import Resize
import torch.nn.functional as F

# Spectral Normalization
def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module

# Self-Attention Mechanism
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

# Improved Generator with Self-Attention
class Generator(nn.Module):
    def __init__(self, input_dim, img_size, img_channels):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(128, 128, 3, stride=1, padding=1)),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True),
            SelfAttention(128),
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(128, 64, 3, stride=1, padding=1)),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),
            SelfAttention(64),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Improved Discriminator with Self-Attention
class Discriminator(nn.Module):
    def __init__(self, img_size, img_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, 16, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            SelfAttention(16),
            spectral_norm(nn.Conv2d(16, 32, 3, 2, 1)),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            spectral_norm(nn.Conv2d(32, 64, 3, 2, 1)),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            SelfAttention(64),
            spectral_norm(nn.Conv2d(64, 128, 3, 2, 1)),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        self.adv_layer = nn.Sequential(nn.Linear(self._get_flatten_size(img_size, img_channels), 1))

    def _get_flatten_size(self, img_size, img_channels):
        with torch.no_grad():
            x = torch.zeros(1, img_channels, img_size, img_size)
            x = self.model(x)
            return x.view(1, -1).size(1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

# Exponential Moving Average (EMA) for Generator
class EMA:
    def __init__(self, model, beta=0.999):
        self.beta = beta
        self.model = model
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters()}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.beta * self.shadow[name] + (1.0 - self.beta) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

# Gradient Penalty
def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.Tensor(np.ones(d_interpolates.shape)).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def get_dataloader(dataset_name, img_size, batch_size):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    elif dataset_name == 'CIFAR-10':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    else:
        raise ValueError("Unsupported dataset. Choose from 'MNIST' or 'CIFAR-10'.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return dataloader

def train_gan(args):
    # Hyperparameters
    latent_dim = args.latent_dim
    img_size = args.img_size
    img_channels = 1 if args.dataset == 'MNIST' else 3
    batch_size = args.batch_size
    learning_rate_G = args.learning_rate_G
    learning_rate_D = args.learning_rate_D
    epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create directories if they don't exist
    os.makedirs("images", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Data Loader
    dataloader = get_dataloader(args.dataset, img_size, batch_size)

    # Initialize models
    generator = Generator(latent_dim, img_size, img_channels).to(device)
    discriminator = Discriminator(img_size, img_channels).to(device)

    # EMA for generator
    ema = EMA(generator)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate_D, betas=(0.5, 0.999))

    # TensorBoard writer
    writer = SummaryWriter()

    # Lists for tracking loss
    g_losses = []
    d_losses = []

    # Training the GAN
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            try:
                # Ground truths with label smoothing
                valid = torch.full((imgs.size(0), 1), 0.9, dtype=torch.float, device=device)
                fake = torch.full((imgs.size(0), 1), 0.1, dtype=torch.float, device=device)
                
                # Configure input
                real_imgs = imgs.to(device)
                
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                
                # Sample noise as generator input
                z = torch.randn((imgs.size(0), latent_dim)).to(device)
                
                # Generate a batch of images
                gen_imgs = generator(z)
                
                # Loss measures generator's ability to fool the discriminator
                g_loss = -torch.mean(discriminator(gen_imgs))
                
                g_loss.backward()
                optimizer_G.step()
                
                # Update EMA
                ema.update()
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()
                
                # Measure discriminator's ability to classify real from generated samples
                real_loss = torch.mean(discriminator(real_imgs))
                fake_loss = torch.mean(discriminator(gen_imgs.detach()))
                gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data, device)
                d_loss = fake_loss - real_loss + 10 * gradient_penalty
                
                d_loss.backward()
                optimizer_D.step()
                
                # Track loss
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                
                # Print progress
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
            except Exception as e:
                print(f"Error during training at epoch {epoch}, batch {i}: {e}")
        
        # Save generated images for each epoch
        save_image(gen_imgs.data[:25], f"images/{epoch}.png", nrow=5, normalize=True)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Generator', g_loss.item(), epoch)
        writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch)
        writer.add_images('Generated Images', gen_imgs.data[:25], epoch, dataformats='NCHW')
        
        # Log model parameters and gradients
        for name, param in generator.named_parameters():
            writer.add_histogram(f'Generator/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Generator/{name}.grad', param.grad, epoch)
        for name, param in discriminator.named_parameters():
            writer.add_histogram(f'Discriminator/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Discriminator/{name}.grad', param.grad, epoch)
        
        # Save model checkpoints
        torch.save(generator.state_dict(), f"checkpoints/generator_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"checkpoints/discriminator_epoch_{epoch}.pth")

    # Apply EMA to generator for final model
    ema.apply_shadow()

    # Save the final models
    torch.save(generator.state_dict(), "generator_final.pth")
    torch.save(discriminator.state_dict(), "discriminator_final.pth")

    # Plot and save the loss curves
    plt.figure()
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/loss_curve.png')
    plt.show()

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--learning_rate_G', type=float, default=0.0001, help='adam: learning rate for generator')
    parser.add_argument('--learning_rate_D', type=float, default=0.0004, help='adam: learning rate for discriminator')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--dataset', type=str, choices=['MNIST', 'CIFAR-10'], default='MNIST', help='dataset to use')
    args = parser.parse_args()

    train_gan(args)
