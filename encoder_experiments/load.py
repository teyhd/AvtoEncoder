import os
import numpy as np
from PIL import Image

from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam


def load_images_from_folder(folder, image_size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).resize(image_size)
            img_array = np.array(img)
            img.save(f'f/{filename}.jpg')
            images.append(img_array)
    return np.array(images)


# Пример использования
folder_path = 'vlad'
image_size = (128, 128)  # Измените размер по вашему желанию
images = load_images_from_folder(folder_path, image_size)

# Нормализуем данные
images = images.astype('float32') / 255.0

# Параметры модели
input_shape = image_size + (3,)  # 3 канала для RGB изображений
latent_dim = 1024  # Размерность латентного пространства

# Encoder
inputs = Input(shape=input_shape)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(inputs)
x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
shape_before_flattening = x.shape[1:]
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# Decoder
decoder_input = Input(shape=(latent_dim,))
x = Dense(np.prod(shape_before_flattening))(decoder_input)
x = Reshape(shape_before_flattening)(x)
x = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

# Модели
encoder = Model(inputs, latent, name='encoder')
decoder = Model(decoder_input, decoded, name='decoder')
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')

# Компиляция и обучение
autoencoder.compile(optimizer=Adam(), loss='mse')
autoencoder.fit(images, images, epochs=150, batch_size=2, validation_split=0.25)

#autoencoder.save('model')
def generate_random_images(decoder, num_images, latent_dim):
    random_latent_vectors = np.random.normal(size=(num_images, latent_dim))
    generated_images = decoder.predict(random_latent_vectors)
    return generated_images


# Пример использования
num_images_to_generate = 90
generated_images = generate_random_images(decoder, num_images_to_generate, latent_dim)

# Преобразование и сохранение изображений
generated_images = (generated_images * 255).astype('uint8')
for i, img in enumerate(generated_images):
    img = Image.fromarray(img)
    img.save(f'output/generated_image_{i}.jpg')

def generate_images_from_folder(folder, encoder, decoder, image_size=(64, 64), noise_factor=0.1, num_images=10):
    images = load_images_from_folder(folder, image_size)

    print(0)

    # Кодирование изображений для получения латентных векторов
    latent_vectors = encoder.predict(images)
    print(1)
    # Генерация новых изображений с добавлением шума к латентным векторам
    generated_images = []
    for i in range(num_images):
        noise = np.random.normal(size=latent_vectors.shape) * noise_factor
        noisy_latent_vectors = latent_vectors + noise
        generated_images_batch = decoder.predict(noisy_latent_vectors)
        generated_images.append(generated_images_batch)

    # Преобразование списка массивов в единый массив
    generated_images = np.vstack(generated_images)

    return generated_images

folder_path = 'NO'
generated_images = generate_images_from_folder(folder_path, encoder, decoder, image_size=(64, 64), noise_factor=0.1, num_images=10)

# Сохранение сгенерированных изображений
for i, img_array in enumerate(generated_images):
    img = Image.fromarray((img_array * 255).astype('uint8'))
    img.save(f"noout/generated_image_{i}.png")

decoder.save('decoder')
encoder.save('encoder')
