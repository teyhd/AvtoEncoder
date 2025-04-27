import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()
        assert img_size % 16 == 0, "IMAGE_SIZE должно быть кратно 16"

        num_layers = int(torch.log2(torch.tensor(img_size // 4)))
        layers = []
        in_channels = 3

        for _ in range(num_layers):
            out_channels = in_channels * 2
            layers.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1))
            layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_channels * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(in_channels * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()
        assert img_size % 16 == 0, "IMAGE_SIZE должно быть кратно 16"

        num_layers = int(torch.log2(torch.tensor(img_size // 4)))
        self.start_channels = 3 * (2 ** num_layers)
        self.fc = nn.Linear(latent_dim, self.start_channels * 4 * 4)

        layers = []
        in_channels = self.start_channels

        for _ in range(num_layers):
            out_channels = in_channels // 2
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1))
            layers.append(nn.ReLU())
            in_channels = out_channels

        # Последний слой заменяем на вывод 3 каналов (RGB) + Sigmoid
        layers[-2] = nn.ConvTranspose2d(in_channels * 2, 3, 4, 2, 1)  # out_channels * 2 -> вернем правильный размер
        layers[-1] = nn.Sigmoid()

        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.start_channels, 4, 4)
        x = self.deconv(x)
        return x

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