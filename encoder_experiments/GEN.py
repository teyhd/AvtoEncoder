import os
import numpy as np
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt
# Функция загрузки изображений из папки
def load_images_from_folder(folder, image_size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).resize(image_size)
            img_array = np.array(img) / 255.0  # Нормализация изображений
            images.append(img_array)
    return np.array(images)

# Функция генерации изображений на основе изображений из папки
def generate_images_from_folder(folder, encoder, decoder, image_size=(128, 128), noise_factor=0.1, num_images=10):
    images = load_images_from_folder(folder, image_size)

    # Проверка наличия достаточного количества изображений
    if len(images) == 0:
        raise ValueError("No images found in the specified folder.")

    # Кодирование изображений для получения латентных векторов
    latent_vectors = encoder.predict(images)
    Image.fromarray((latent_vectors * 255).astype('uint8')).save(f"noout/000.png")
    # Генерация новых изображений с добавлением шума к латентным векторам
    generated_images = []
    for i in range(num_images):
        noise = np.random.normal(size=latent_vectors.shape) * noise_factor
        noisy_latent_vectors = latent_vectors + noise
        generated_images_batch = decoder.predict(noisy_latent_vectors)
        generated_images.append(generated_images_batch)

    # Преобразование списка массивов в единый массив
    generated_images = np.vstack(generated_images)

    mse = np.mean((images - generated_images[:len(images)])**2)
    print(f"Mean Squared Error between original and generated images: {mse}")

    # Пример визуализации оригинальных и сгенерированных изображений

    combined_images = []
    for i in range(len(images)):
        original_img = (images[i] * 255).astype('uint8')
        generated_img = (generated_images[i] * 255).astype('uint8')

        combined_img = np.hstack((original_img, generated_img))
        combined_images.append(combined_img)


        img = Image.fromarray(combined_img)
        img.save(f"noout/generated_images_{i}.png")

    for i in range(min(5, len(images))):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title('Original')
        plt.axis('off')

        plt.subplot(2, 5, i + 6)
        plt.imshow(generated_images[i])
        plt.title('Generated')
        plt.axis('off')

        plt.subplot(2, 5, i + 6)
        plt.imshow(combined_images[i])
        plt.title('Mixed')
        plt.axis('off')
    plt.show()


    return generated_images

# Пример использования функции
folder_path = 'NO'
# Предполагаем, что encoder и decoder уже обучены и загружены
# Например, если у вас сохранены модели энкодера и декодера, вы можете их загрузить так:
encoder = load_model('encoder')
decoder = load_model('decoder')
noise_factor=0.1
num_images=2
# Генерация новых изображений
generated_images = generate_images_from_folder(folder_path, encoder, decoder, image_size=(128, 128), noise_factor=noise_factor, num_images=num_images)
generated_images = (generated_images * 255).astype('uint8')
# Сохранение сгенерированных изображений
for i, img_array in enumerate(generated_images):
    img = Image.fromarray(img_array)
    img.save(f"noout/generated_image_{i}.png")
