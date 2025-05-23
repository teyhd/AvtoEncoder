# --- Исправленная структура VAE с правильным масштабированием ---

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

# --- Encoder ---
class Encoder(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()
        assert img_size >= 64 and img_size % 16 == 0, "IMAGE_SIZE должно быть кратно 16 и >= 64"

        base_channels = 32
        layers = []
        in_channels = 3

        self.num_blocks = int(torch.log2(torch.tensor(img_size // 4)).item())

        for _ in range(self.num_blocks):
            out_channels = base_channels
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
            base_channels *= 2

        self.conv = nn.Sequential(*layers)
        self.final_spatial = img_size // (2 ** self.num_blocks)
        self.fc_mu = nn.Linear(in_channels * self.final_spatial * self.final_spatial, latent_dim)
        self.fc_logvar = nn.Linear(in_channels * self.final_spatial * self.final_spatial, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# --- Decoder ---
class Decoder(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()
        assert img_size >= 64 and img_size % 16 == 0, "IMAGE_SIZE должно быть кратно 16 и >= 64"

        self.img_size = img_size
        self.num_blocks = int(torch.log2(torch.tensor(img_size // 4)).item())
        self.final_spatial = img_size // (2 ** self.num_blocks)
        self.start_channels = 32 * (2 ** self.num_blocks)

        self.fc = nn.Linear(latent_dim, self.start_channels * self.final_spatial * self.final_spatial)

        layers = []
        in_channels = self.start_channels
        for idx in range(self.num_blocks - 1):
            out_channels = in_channels // 2
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # Финальный слой отдельно
        layers.append(nn.ConvTranspose2d(in_channels, 3, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Sigmoid())

        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.start_channels, self.final_spatial, self.final_spatial)
        x = self.deconv(x)
        return x

# --- Полная модель VAE ---
class VAE(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()
        self.encoder = Encoder(latent_dim, img_size)
        self.decoder = Decoder(latent_dim, img_size)
        self.img_size = img_size

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        # Обеспечиваем совпадение размера реконструкции с оригиналом
        if recon_x.shape[2:] != x.shape[2:]:
            recon_x = torch.nn.functional.interpolate(recon_x, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return recon_x, mu, logvar

# --- Перцептуальная функция потерь VGG ---
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device='cuda', layers=(3, 8, 15), normalize=True):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False

        self.blocks = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:layer+1]).to(device) for layer in layers
        ])
        self.normalize = normalize

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        if self.normalize:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += nn.functional.mse_loss(x, y)
        return loss
