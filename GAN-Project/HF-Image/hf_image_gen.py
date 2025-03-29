import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import numpy as np
import os
import argparse
from torch.utils.tensorboard import SummaryWriter

# Import CLIP for text-to-image generation (if available)
try:
    import clip
except ImportError:
    print("CLIP library not found, text-to-image generation will not be available. Install it using 'pip install git+https://github.com/openai/CLIP.git'.")

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = F.normalize(torch.matmul(w.view(height, -1).data.t(), u.data), dim=0)
            u.data = F.normalize(torch.matmul(w.view(height, -1).data, v.data), dim=0)

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data, dim=0)
        v.data = F.normalize(v.data, dim=0)
        w_bar = nn.Parameter(w.data)

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

        del self.module._parameters[self.name]

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class Generator(nn.Module):
    def __init__(self, z_dim, img_size, num_classes, ch=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.img_size = img_size
        self.num_classes = num_classes
        self.ch = ch

        self.fc = nn.Sequential(
            nn.Linear(z_dim + num_classes, 4 * 4 * 16 * ch),
            nn.BatchNorm1d(4 * 4 * 16 * ch),
            nn.ReLU()
        )
        self.deconv_blocks = nn.ModuleList([
            nn.Sequential(
                SpectralNorm(nn.ConvTranspose2d(16 * ch, 8 * ch, kernel_size=4, stride=2, padding=1)),
                nn.BatchNorm2d(8 * ch),
                nn.ReLU()
            ),
            nn.Sequential(
                SpectralNorm(nn.ConvTranspose2d(8 * ch, 4 * ch, kernel_size=4, stride=2, padding=1)),
                nn.BatchNorm2d(4 * ch),
                nn.ReLU()
            ),
            nn.Sequential(
                SpectralNorm(nn.ConvTranspose2d(4 * ch, 2 * ch, kernel_size=4, stride=2, padding=1)),
                nn.BatchNorm2d(2 * ch),
                nn.ReLU()
            ),
            nn.Sequential(
                SpectralNorm(nn.ConvTranspose2d(2 * ch, ch, kernel_size=4, stride=2, padding=1)),
                nn.BatchNorm2d(ch),
                nn.ReLU()
            ),
            nn.Sequential(
                SpectralNorm(nn.ConvTranspose2d(ch, 3, kernel_size=4, stride=2, padding=1)),
                nn.Tanh()
            )
        ])

    def forward(self, z, labels):
        z = torch.cat([z, labels], dim=1)
        out = self.fc(z).view(-1, 16 * self.ch, 4, 4)
        for block in self.deconv_blocks:
            out = block(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, img_size, num_classes, ch=64):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.ch = ch

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                SpectralNorm(nn.Conv2d(3, ch, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                SpectralNorm(nn.Conv2d(ch, 2 * ch, kernel_size=4, stride=2, padding=1)),
                nn.BatchNorm2d(2 * ch),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                SpectralNorm(nn.Conv2d(2 * ch, 4 * ch, kernel_size=4, stride=2, padding=1)),
                nn.BatchNorm2d(4 * ch),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                SpectralNorm(nn.Conv2d(4 * ch, 8 * ch, kernel_size=4, stride=2, padding=1)),
                nn.BatchNorm2d(8 * ch),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                SpectralNorm(nn.Conv2d(8 * ch, 16 * ch, kernel_size=4, stride=2, padding=1)),
                nn.BatchNorm2d(16 * ch),
                nn.LeakyReLU(0.2)
            )
        ])
        
        self.fc_input_dim = 16 * ch * (img_size // 32) * (img_size // 32) + num_classes
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        out = x
        for block in self.conv_blocks:
            out = block(out)
        out = out.view(out.size(0), -1)
        labels = labels.view(labels.size(0), -1)  # Flatten the labels
        out = torch.cat([out, labels], dim=1)
        out = self.fc(out)
        return out

def train_gan(generator, discriminator, dataloader, num_epochs, z_dim, num_classes, lr_g, lr_d, beta1, beta2, log_dir='runs', image_word=None, device='cpu'):
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))

    fixed_noise = torch.randn(64, z_dim, device=device)
    fixed_labels = torch.eye(num_classes)[torch.randint(0, num_classes, (64,))].to(device)

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        for i, (real_images, labels) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = torch.eye(num_classes)[labels].to(device)

            # Train Discriminator
            optimizer_d.zero_grad()
            z = torch.randn(batch_size, z_dim, device=device)
            fake_images = generator(z, labels)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            real_loss = criterion(discriminator(real_images, labels), real_labels)
            fake_loss = criterion(discriminator(fake_images.detach(), labels), fake_labels)
            d_loss = real_loss + fake_loss

            d_loss.backward()
            optimizer_d.step()

            # Log discriminator gradients
            for name, param in discriminator.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'Discriminator_Gradients/{name}', param.grad, epoch * len(dataloader) + i)

            # Train Generator
            optimizer_g.zero_grad()
            g_loss = criterion(discriminator(fake_images, labels), real_labels)

            g_loss.backward()
            optimizer_g.step()

            # Log generator gradients
            for name, param in generator.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'Generator_Gradients/{name}', param.grad, epoch * len(dataloader) + i)

            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')
                writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(dataloader) + i)

        # Log histograms of the weights
        for name, param in discriminator.named_parameters():
            writer.add_histogram(f'Discriminator_Weights/{name}', param, epoch)
        for name, param in generator.named_parameters():
            writer.add_histogram(f'Generator_Weights/{name}', param, epoch)

        with torch.no_grad():
            fake_images = generator(fixed_noise, fixed_labels).detach().cpu()
            img_grid = make_grid(fake_images, normalize=True, scale_each=True)
            writer.add_image('Generated Images', img_grid, epoch)

        writer.flush()

    writer.close()

def truncation_trick(z, threshold):
    norm = torch.norm(z, dim=1, keepdim=True)
    z = torch.where(norm < threshold, z, z / norm * threshold)
    return z

def main():
    parser = argparse.ArgumentParser(description='Train a GAN on CIFAR-10 dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr_g', type=float, default=0.0002, help='learning rate for generator (default: 0.0002)')
    parser.add_argument('--lr_d', type=float, default=0.0002, help='learning rate for discriminator (default: 0.0002)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer (default: 0.999)')
    parser.add_argument('--log_dir', type=str, default='runs/cifar10_gan', help='log directory for TensorBoard (default: runs/cifar10_gan)')
    parser.add_argument('--image_word', type=str, help='word to generate image from (optional)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = 128  # Image size for CIFAR-10
    z_dim = 120
    num_classes = 10  # CIFAR-10 has 10 classes

    transform = transforms.Compose([
        transforms.Resize(img_size),  # Resize images to 128x128
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    
    dataset = datasets.CIFAR10(root='data', download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    generator = Generator(z_dim=z_dim, img_size=img_size, num_classes=num_classes)
    discriminator = Discriminator(img_size=img_size, num_classes=num_classes)

    if not os.path.exists('samples'):
        os.makedirs('samples')

    if args.image_word and 'clip' in globals():
        model, preprocess = clip.load("ViT-B/32", device=device)
        text = clip.tokenize([args.image_word]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text).to(device)
        generator.word_embedding = text_features
    else:
        generator.word_embedding = None

    train_gan(generator, discriminator, dataloader, args.epochs, z_dim, num_classes, args.lr_g, args.lr_d, args.beta1, args.beta2, args.log_dir, args.image_word, device)

if __name__ == '__main__':
    main()
