import torch
import os
import numpy as np
from sklearn.decomposition import PCA
from torchvision import transforms
from PIL import Image
import importlib

# --- Параметры ---
INPUT_DIR = './data/Vlad'
MODEL_PATH = './models/VAE_model.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
module = importlib.import_module(f'VAE.vae_v{checkpoint['v']}')
VAE = module.VAE

print(checkpoint['v'])
BATCH_SIZE = checkpoint['BATCH_SIZE']
LATENT_DIM = checkpoint['LATENT_DIM']
IMAGE_SIZE = checkpoint['IMAGE_SIZE']
# --- Подготовка ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

VAE = VAE(LATENT_DIM, IMAGE_SIZE).to(DEVICE)
if MODEL_PATH:
    VAE.load_state_dict(checkpoint['model'])
    print("Loaded model from", MODEL_PATH)
VAE.eval()

# --- Загрузка всех фото одного человека ---
latent_vectors = []

for img_name in os.listdir(INPUT_DIR):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img = Image.open(os.path.join(INPUT_DIR, img_name)).convert('RGB')
        img = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            mu, logvar = VAE.encoder(img)
            z = VAE.reparameterize(mu, logvar)
            latent_vectors.append(z.squeeze(0).cpu().numpy())

latent_vectors = np.stack(latent_vectors)

# --- PCA по латентным векторам ---
pca = PCA(n_components=3)  # Можно больше, но 2 достаточно для анализа
pca.fit(latent_vectors)
components = pca.components_

# --- Получение векторов крайних точек ---
projections = latent_vectors @ components[0]  # Проекция на первую компоненту
z_min = latent_vectors[np.argmin(projections)]
z_max = latent_vectors[np.argmax(projections)]

# --- Построение "эмоционального вектора" ---
emotion_shift = z_max - z_min

# --- Применение к другому лицу ---
def apply_emotion_shift(img_path, shift, strength=1.0):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        mu, logvar = VAE.encoder(img)
        z = VAE.reparameterize(mu, logvar)
        z_new = z + torch.from_numpy(shift).to(DEVICE) * strength
        generated = VAE.decoder(z_new.unsqueeze(0))
        
    return generated

# --- Пример применения ---
result = apply_emotion_shift('./input/1.jpg', emotion_shift, strength=0.5)

# --- Сохранение результата ---
import torchvision.utils as vutils
vutils.save_image(result.cpu(), 'shifted_face.png', normalize=True)
