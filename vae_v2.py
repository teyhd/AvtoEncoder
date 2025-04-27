# --- Улучшенная архитектура VAE с поддержкой высокого качества изображений ---

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

# --- Расширенный Encoder ---
class Encoder(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()
        assert img_size >= 64 and img_size % 16 == 0, "IMAGE_SIZE должно быть кратно 16 и >= 64"

        # Стартовое количество каналов
        base_channels = 32
        layers = []
        in_channels = 3

        # Фиксированное количество свёрток: 5 штук
        for _ in range(5):
            out_channels = base_channels
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
            base_channels *= 2  # Удвоение каналов

        self.conv = nn.Sequential(*layers)
        final_size = img_size // (2 ** 5)  # После 5 слоёв уменьшается в 32 раза
        self.fc_mu = nn.Linear(in_channels * final_size * final_size, latent_dim)
        self.fc_logvar = nn.Linear(in_channels * final_size * final_size, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# --- Расширенный Decoder ---
class Decoder(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()
        assert img_size >= 64 and img_size % 16 == 0, "IMAGE_SIZE должно быть кратно 16 и >= 64"

        base_channels = 32 * (2 ** 5)  # Должно совпадать с последним каналом энкодера
        final_size = img_size // (2 ** 5)

        self.fc = nn.Linear(latent_dim, base_channels * final_size * final_size)

        layers = []
        in_channels = base_channels
        for _ in range(5):
            out_channels = in_channels // 2
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        layers[-2] = nn.ConvTranspose2d(in_channels * 2, 3, kernel_size=4, stride=2, padding=1)
        layers[-1] = nn.Sigmoid()

        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        final_size = int((x.size(-1) / (32 * 2 ** 5)) ** 0.5)
        x = x.view(-1, 32 * (2 ** 5), final_size, final_size)
        x = self.deconv(x)
        return x

# --- Полная модель VAE ---
class VAE(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()
        self.encoder = Encoder(latent_dim, img_size)
        self.decoder = Decoder(latent_dim, img_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
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
