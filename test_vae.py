import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from vae_v2 import VAE, VGGPerceptualLoss
# Используем исправленный VAE из описания выше

# --- Параметры для теста ---
BATCH_SIZE = 8
LATENT_DIM = 512
IMAGE_SIZE = 256

# --- Тест модели ---
vae = VAE(latent_dim=LATENT_DIM, img_size=IMAGE_SIZE)

# случайный вход
x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
recon_x, mu, logvar = vae(x)

# Проверка размеров
assert recon_x.shape == x.shape, f"Размер реконструированного изображения неправильный: {recon_x.shape} != {x.shape}"

# Визуализация
grid = make_grid(recon_x, nrow=4)
plt.figure(figsize=(10,10))
plt.axis('off')
plt.title('Реконструированные изображения')
plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
plt.show()

