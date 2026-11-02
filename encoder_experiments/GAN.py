import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import LeakyReLU, UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def load_images(path, image_size=(128, 128)):
    images = []
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file))
        if img is not None:
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return np.array(images) / 127.5 - 1.0

# Загрузка ваших фотографий
my_images = load_images("vlad")

# Загрузка фотографий, на которые нужно заменить лицо
target_images = load_images("NO")

def build_generator():
    model = Sequential()
    model.add(Dense(128 * 32 * 32, activation="relu", input_dim=100))
    model.add(Reshape((32, 32, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    noise = Input(shape=(100,))
    img = model(noise)
    return Model(noise, img)

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(128, 128, 3), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    img = Input(shape=(128, 128, 3))
    validity = model(img)
    return Model(img, validity)

def train_gan(epochs, batch_size=128, save_interval=50):
    half_batch = int(batch_size / 2)

    # Загрузка моделей
    optimizer = Adam(0.0002, 0.5)
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    generator = build_generator()

    z = Input(shape=(100,))
    img = generator(z)
    discriminator.trainable = False
    validity = discriminator(img)

    combined = Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    for epoch in range(epochs):
        # Обучение дискриминатора
        idx = np.random.randint(0, my_images.shape[0], half_batch)
        imgs = my_images[idx]
        noise = np.random.normal(0, 1, (half_batch, 100))
        gen_imgs = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Обучение генератора
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.array([1] * batch_size)
        g_loss = combined.train_on_batch(noise, valid_y)

        print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {d_loss[1]}] [G loss: {g_loss}]")

        if epoch % save_interval == 0:
            save_images(epoch, generator)
            save_models(generator, discriminator, epoch, save_dir="GAN")

def save_images(epoch, generator):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(f"images/mnist_{epoch}.png")
    plt.close()

def save_models(generator, discriminator, epoch, save_dir="saved_models"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    generator.save(os.path.join(save_dir, f"generator_epoch_{epoch}.h5"))
    discriminator.save(os.path.join(save_dir, f"discriminator_epoch_{epoch}.h5"))
    print(f"Models saved at epoch {epoch}")


train_gan(epochs=25000, batch_size=32, save_interval=1000)
