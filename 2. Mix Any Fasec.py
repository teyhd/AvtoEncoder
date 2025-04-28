# --- Скрипт: Расширенное смешивание лиц с циклической GIF-анимацией и логами работы ---
import numpy as np
import torch
import torchvision.transforms as transforms
import imageio
import os
from PIL import Image
import importlib

MODEL_PATH = './models/vae_model_V2.pth'#  vae_model_V2
INPUT_DIR = 'input'
GIF_PATH = './mix/face_morph.gif'
CROP_PERCENT = 1
MORPH_STEPS = 50
FPS = 25

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
module = importlib.import_module(f'VAE.vae_v{checkpoint['v']}')
VAE = module.VAE
print(checkpoint['v'])
BATCH_SIZE = checkpoint['BATCH_SIZE']
LATENT_DIM = checkpoint['LATENT_DIM']
IMAGE_SIZE = checkpoint['IMAGE_SIZE']

# --- Функция Slerp ---
def slerp(val, low, high):
    low_norm = low / low.norm(dim=-1, keepdim=True)
    high_norm = high / high.norm(dim=-1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(-1))
    so = torch.sin(omega)
    so = torch.where(so == 0, torch.tensor(1e-6, device=low.device), so)
    return (torch.sin((1.0 - val) * omega) / so).unsqueeze(-1) * low + (torch.sin(val * omega) / so).unsqueeze(-1) * high

# --- Подготовка директорий ---
os.makedirs(INPUT_DIR, exist_ok=True)

# --- Трансформация изображений ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# --- Функция центрированного кропа ---
def center_crop(img, crop_percent=CROP_PERCENT):
    width, height = img.size
    new_width = int(width * crop_percent)
    new_height = int(height * crop_percent)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return img.crop((left, top, right, bottom))

# --- Загрузка изображения ---
def load_image(path):
    img = Image.open(path).convert('RGB')
    img = center_crop(img)
    img = transform(img)
    return img.unsqueeze(0).to(DEVICE)

# --- Загрузка модели ---
vae = VAE(LATENT_DIM, IMAGE_SIZE).to(DEVICE)
if MODEL_PATH:
    vae.load_state_dict(checkpoint['model'])
    print(f"[INFO] Модель загружена из {MODEL_PATH}")
vae.eval()

# --- Загрузка изображений ---
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
image_files.sort()
if len(image_files) < 2:
    raise ValueError("[ERROR] Нужно минимум два изображения в папке input!")

images = [load_image(os.path.join(INPUT_DIR, img)) for img in image_files]
print(f"[INFO] Загружено {len(images)} изображений.")

# --- Кодирование всех изображений ---
print("[INFO] Кодирование изображений в латентное пространство...")
zs = []
with torch.no_grad():
    for img in images:
        mu, logvar = vae.encoder(img)
        z = vae.reparameterize(mu, logvar)
        zs.append(z)

# --- Генерация кадров ---
print("[INFO] Генерация кадров морфинга...")
frames = []
with torch.no_grad():
    for i in range(len(zs)):
        z1 = zs[i]
        z2 = zs[(i + 1) % len(zs)]  # Зацикливаем

        for step in range(MORPH_STEPS):
            alpha = step / (MORPH_STEPS - 1)
            z_mix = slerp(alpha, z1, z2)
            mixed_face = vae.decoder(z_mix)

            frame = mixed_face.squeeze(0).cpu()
           # frame = frame * 0.5 + 0.5  # Денормализация
            #frame = (frame + 1) / 2
            frame = frame.clamp(0, 1)

            frame = transforms.ToPILImage()(frame)
            frames.append(np.array(frame))

        print(f"[INFO] Сгенерирован переход {i + 1}/{len(zs)}")

# --- Сохранение GIF ---
print("[INFO] Сохранение GIF анимации...")
imageio.mimsave(GIF_PATH, frames, fps=FPS)
print(f"[SUCCESS] GIF сохранен: {GIF_PATH}")
