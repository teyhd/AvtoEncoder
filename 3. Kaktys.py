# --- Проект: Смешивание лица и кактуса через автоэнкодер VAE ---
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import importlib
from torchvision.utils import save_image

# --- Параметры ---
MODEL_PATH = './models/vae_model.pth'
FACE_PATH = './input/face.jpg'
CACTUS_PATH = './input/cactus (3).jpg'
OUTPUT_PATH = 'vae_blended_face.png'
ALPHA_BLEND = 0.55  # Доля влияния кактуса

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Загрузка модели VAE ---
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
module = importlib.import_module(f'VAE.vae_v{checkpoint["v"]}')
VAE = module.VAE

vae = VAE(checkpoint['LATENT_DIM'], checkpoint['IMAGE_SIZE']).to(DEVICE)
vae.load_state_dict(checkpoint['model'])
vae.eval()

# --- Трансформации ---
transform = transforms.Compose([
    transforms.Resize((checkpoint['IMAGE_SIZE'], checkpoint['IMAGE_SIZE'])),
    transforms.ToTensor()
])

# --- Загрузка и подготовка изображений ---
def load_image(path):
    img = Image.open(path).convert('RGB')
    img = transform(img)
    return img.unsqueeze(0).to(DEVICE)

face_img = load_image(FACE_PATH)
cactus_img = load_image(CACTUS_PATH)

# --- Кодирование в латентное пространство ---
with torch.no_grad():
    mu_face, logvar_face = vae.encoder(face_img)
    mu_cactus, logvar_cactus = vae.encoder(cactus_img)

    z_face = vae.reparameterize(mu_face, logvar_face)
    z_cactus = vae.reparameterize(mu_cactus, logvar_cactus)

    # Смешивание латентных векторов
    z_blend = z_face + ALPHA_BLEND * (z_cactus - z_face)

    # Декодирование обратно в изображение
    blended_img = vae.decoder(z_blend)

# --- Сохранение результата ---
#os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
save_image(blended_img, OUTPUT_PATH, normalize=True)
print(f"Сохранено смешанное изображение через VAE: {OUTPUT_PATH}")
