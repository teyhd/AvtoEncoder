# --- Скрипт: смешивание двух лиц и создание GIF анимации с авто-кропом ---
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from PIL import Image
import importlib

# --- Параметры ---
MODEL_PATH = './models/vae_model.pth'
MORPH_STEPS = 100
INPUT_DIR = 'input'
SAVE_DIR = './mix/morph_frames'
GIF_PATH = './mix/face_morph.gif'
CROP_PERCENT = 0.999  # Обрезаем центральную часть изображения (80%)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
module = importlib.import_module(f'VAE.vae_v{checkpoint['v']}')
VAE = module.VAE

print(checkpoint['v'])
BATCH_SIZE = checkpoint['BATCH_SIZE']
LATENT_DIM = checkpoint['LATENT_DIM']
IMAGE_SIZE = checkpoint['IMAGE_SIZE']

# Получаем список файлов и фильтруем только картинки
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
image_files.sort()  # Опционально: сортировка для стабильности

# --- Подготовка ---
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])
# Инициализация модели
vae = VAE(LATENT_DIM, IMAGE_SIZE).to(DEVICE)
if MODEL_PATH:
    vae.load_state_dict(checkpoint['model'])
    print("Loaded model from", MODEL_PATH)

# --- Функция отображения изображения ---
def imshow(img):
    img = img.cpu().detach()
    img = img * 0.5 + 0.5  # Если были нормализации - восстановить
    npimg = img.numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

# --- Функция для центрированного кропа ---
def center_crop(img, crop_percent=CROP_PERCENT):
    width, height = img.size
    new_width = int(width * crop_percent)
    new_height = int(height * crop_percent)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return img.crop((left, top, right, bottom))

# --- Загрузка изображений ---
def load_image(path):
    img = Image.open(path).convert('RGB')
    img = center_crop(img)
    img = transform(img)
    return img.unsqueeze(0).to(DEVICE)


# Загружаем первые две картинки
img1 = load_image(os.path.join(INPUT_DIR, image_files[0]))
img2 = load_image(os.path.join(INPUT_DIR, image_files[1]))

# --- Получение латентных векторов ---
vae.eval()
with torch.no_grad():
    mu1, logvar1 = vae.encoder(img1)
    mu2, logvar2 = vae.encoder(img2)

    z1 = vae.reparameterize(mu1, logvar1)
    z2 = vae.reparameterize(mu2, logvar2)

# --- Интерполяция и создание кадров ---
frames = []
with torch.no_grad():
    for step in range(MORPH_STEPS):
        alpha = step / (MORPH_STEPS - 1)
        z_mix = (1 - alpha) * z1 + alpha * z2
        mixed_face = vae.decoder(z_mix)

        frame_path = os.path.join(SAVE_DIR, f'frame_{step:03d}.png')
        vutils.save_image(mixed_face.cpu(), frame_path, nrow=1, normalize=True)

        frame = imageio.imread(frame_path)
        frames.append(frame)

# --- Сохранение анимации ---
imageio.mimsave(GIF_PATH, frames, fps=10)
print(f"GIF анимация сохранена в {GIF_PATH}")

# --- Отобразить последний кадр ---
#imshow(torchvision.utils.make_grid(mixed_face.cpu()))
