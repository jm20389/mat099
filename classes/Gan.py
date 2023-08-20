import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
from skimage.io import imread

class GanBuilder:
    """
    Adapted from the Udemy Course - Deep Learning: Advanced Computer Vision (GANs, SSD, +More!) by The Lazy Programmer
    """
    def __init__(self,
                 X_train, Y_train, X_test, Y_test,
                 train_dir =     './output/gan_images/',
                 batch_size =    32,
                 epochs =        30000,
                 sample_period = 200,
                 latent_dim =    100,
                 ):

        X_train, X_test = X_train / 255.0 * 2 - 1, X_test / 255.0 * 2 - 1

        self.batch_size =    batch_size
        self.epochs =        epochs
        self.sample_period = sample_period
        self.latent_dim =    latent_dim

        self.Y_train =  Y_train
        self.Y_test =   Y_test

        self.N, self.H, self.W = X_train.shape
        self.D = self.H * self.W

        self.X_train = X_train.reshape(-1, self.D)
        self.X_test =  X_test.reshape(-1, self.D)

        self.train_dir = train_dir if train_dir[-1] == '/' else train_dir + '/'
        os.makedirs(self.train_dir, exist_ok=True)

    def build_generator(self):
        i = Input(shape=(self.latent_dim,))
        x = Dense(256, activation=LeakyReLU(alpha=0.2))(i)
        x = BatchNormalization(momentum=0.7)(x)
        x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
        x = BatchNormalization(momentum=0.7)(x)
        x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
        x = BatchNormalization(momentum=0.7)(x)
        x = Dense(self.D, activation='tanh')(x)
        model = Model(i, x)
        self.generator = model
        return model

    def build_discriminator(self):
        i = Input(shape=(self.D))
        x = Dense(512, activation=LeakyReLU(alpha=0.2))(i)
        x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(i, x)
        self.discriminator = model
        return model

    def compile(self):
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=Adam(0.0002, 0.5),
            metrics=['accuracy'])

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False
        fake_pred = self.discriminator(img)
        self.combined_model = Model(z, fake_pred)
        self.combined_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
        print('Compile completed')
        return None

    def train(self):
        ones =  np.ones(self.batch_size)
        zeros = np.zeros(self.batch_size)

        d_losses = []
        g_losses = []

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        for epoch in range(self.epochs):
            # Select a random batch of images
            idx = np.random.randint(0, self.X_train.shape[0], self.batch_size)
            real_imgs = self.X_train[idx]

            # Generate fake images
            noise = np.random.randn(self.batch_size, self.latent_dim)
            fake_imgs = self.generator.predict(noise)

            # Train the discriminator
            # both loss and accuracy are returned
            d_loss_real, d_acc_real = self.discriminator.train_on_batch(real_imgs, ones)
            d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(fake_imgs, zeros)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_acc  = 0.5 * (d_acc_real + d_acc_fake)

            noise = np.random.randn(self.batch_size, self.latent_dim)
            g_loss = self.combined_model.train_on_batch(noise, ones)

            # do it again!
            noise = np.random.randn(self.batch_size, self.latent_dim)
            g_loss = self.combined_model.train_on_batch(noise, ones)

            # Save the losses
            d_losses.append(d_loss)
            g_losses.append(g_loss)

            if epoch % 100 == 0:
                print(f"epoch: {epoch+1}/{self.epochs}, d_loss: {d_loss:.2f}, \
                d_acc: {d_acc:.2f}, g_loss: {g_loss:.2f}")

            if epoch % self.sample_period == 0:
                self.sample_images(epoch)

        self.d_losses = d_losses
        self.g_losses = g_losses
        print('Train completed')
        return None

    def sample_images(self, epoch):
        rows, cols = 5, 5
        noise = np.random.randn(rows * cols, self.latent_dim)
        imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        imgs = 0.5 * imgs + 0.5

        fig, axs = plt.subplots(rows, cols)
        idx = 0
        for i in range(rows):
            for j in range(cols):
                axs[i,j].imshow(imgs[idx].reshape(self.H, self.W), cmap='gray')
                axs[i,j].axis('off')
                idx += 1
        fig.savefig(self.train_dir+"%d.png" % epoch)
        plt.close()

        return None

