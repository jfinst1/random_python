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
import pennylane as qml
from pennylane import numpy as np
from scipy.stats import entropy
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torchvision.transforms import Resize
import torch.nn.functional as F

# Quantum layer definition
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev, interface='torch')
def quantum_layer(inputs):
    for i in range(4):
        qml.RX(inputs[i], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires[2, 3])
    qml.RY(np.pi / 2, wires=0)
    qml.RY(np.pi / 2, wires=1)
    qml.RY(np.pi / 2, wires=2)
    qml.RY(np.pi / 2, wires=3)
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

class HybridQuantumLayer(nn.Module):
    def __init__(self):
        super(HybridQuantumLayer, self).__init__()
        self.linear = nn.Linear(4, 4)
    
    def forward(self, x):
        x = self.linear(x)
        x = quantum_layer(x)
        return torch.tensor(x, dtype=torch.float32)

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
        return out  # Return only the tensor

class Generator(nn.Module):
    def __init__(self, input_dim, img_size):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True),
            SelfAttention(128),  # Self-Attention layer
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),
            SelfAttention(64),  # Self-Attention layer
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            SelfAttention(16),  # Self-Attention layer
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            SelfAttention(32),  # Self-Attention layer
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        # Compute the size of the flattened layer dynamically
        self.adv_layer = nn.Sequential(nn.Linear(self._get_flatten_size(img_size), 1))

    def _get_flatten_size(self, img_size):
        with torch.no_grad():
            x = torch.zeros(1, 1, img_size, img_size)
            x = self.model(x)
            return x.view(1, -1).size(1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

def gradient_penalty(discriminator, real_imgs, fake_imgs, device):
    alpha = torch.randn((real_imgs.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_imgs + ((1 - alpha) * fake_imgs)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones((real_imgs.size(0), 1), device=device, requires_grad=False)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def calculate_fid(real_images, fake_images):
    real_images = real_images.detach().cpu().numpy()
    fake_images = fake_images.detach().cpu().numpy()

    # Reshape images to (num_samples, num_features)
    real_images = real_images.reshape(real_images.shape[0], -1)
    fake_images = fake_images.reshape(fake_images.shape[0], -1)
    
    mu_real = np.mean(real_images, axis=0)
    sigma_real = np.cov(real_images, rowvar=False)
    mu_fake = np.mean(fake_images, axis=0)
    sigma_fake = np.cov(fake_images, rowvar=False)
    
    ssdiff = np.sum((mu_real - mu_fake) ** 2.0)
    covmean = sqrtm(sigma_real.dot(sigma_fake))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return fid

def inception_score(images, inception_model, device, resize=True, splits=10):
    N = len(images)
    
    if resize:
        up = Resize((299, 299), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        images = [up(image) for image in images]
    
    # Convert grayscale images to RGB
    images = [image.repeat(3, 1, 1) if image.size(0) == 1 else image for image in images]
    
    images = torch.stack(images).to(device)
    batch_size = images.size(0)
    
    def get_pred(x):
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
    
    preds = np.zeros((N, 1000))
    
    for i in range(0, N, batch_size):
        batch = images[i:i+batch_size]
        batch_size_i = batch.size(0)
        preds[i:i+batch_size_i] = get_pred(batch)
    
    split_scores = []
    
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)

def train_gan(args):
    # Hyperparameters
    latent_dim = args.latent_dim
    img_size = args.img_size
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    lambda_gp = 10
    n_critic = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create directories if they don't exist
    os.makedirs("images", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Data Loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize models
    generator = Generator(latent_dim, img_size).to(device)
    discriminator = Discriminator(img_size).to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Pre-trained Inception model for FID and IS
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    # TensorBoard writer
    writer = SummaryWriter()

    # Lists for tracking loss
    g_losses = []
    d_losses = []

    # Training the GAN
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            
            # Ground truths
            valid = torch.ones((imgs.size(0), 1), requires_grad=False).to(device)
            fake = torch.zeros((imgs.size(0), 1), requires_grad=False).to(device)
            
            # Configure input
            real_imgs = imgs.to(device)

            # Add noise to discriminator inputs
            real_imgs += 0.1 * torch.randn_like(real_imgs)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Sample noise as generator input
            z = torch.randn((imgs.size(0), latent_dim)).to(device)
            
            # Generate a batch of images
            gen_imgs = generator(z)
            
            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(gen_imgs.detach())
            # Gradient penalty
            gp = gradient_penalty(discriminator, real_imgs, gen_imgs, device)
            # Adversarial loss
            w_dist = real_validity.mean() - fake_validity.mean()
            d_loss = -w_dist + lambda_gp * gp
            
            d_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)  # Gradient clipping
            optimizer_D.step()

            # Train the generator every n_critic steps
            if i % n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                
                # Generate a batch of images
                gen_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                g_loss = -torch.mean(discriminator(gen_imgs))
                
                g_loss.backward()
                nn.utils.clip_grad_norm_(generator.parameters(), 1.0)  # Gradient clipping
                optimizer_G.step()
                
                # Track loss
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                
                # Print progress
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
        
        # Save generated images for each epoch
        save_image(gen_imgs.data[:25], f"images/{epoch}.png", nrow=5, normalize=True)
        
        # Compute FID and IS
        if epoch % 10 == 0:
            real_images = real_imgs[:batch_size]
            fake_images = gen_imgs[:batch_size]
            fid = calculate_fid(real_images, fake_images)
            is_mean, is_std = inception_score(fake_images, inception_model, device)
            writer.add_scalar('FID', fid, epoch)
            writer.add_scalar('Inception Score Mean', is_mean, epoch)
            writer.add_scalar('Inception Score Std', is_std, epoch)
            print(f"FID: {fid}, Inception Score: {is_mean} ± {is_std}")

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
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs of training')
    args = parser.parse_args()

    train_gan(args)
