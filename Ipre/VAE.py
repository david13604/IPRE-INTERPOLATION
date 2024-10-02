import tensorflow as tf
from keras import layers, models
import numpy as np
import librosa

# Cargar datos de audio
def load_audio(file_path):
    audio, _ = librosa.load(file_path, sr=22050)  # Cambia la tasa de muestreo según tus necesidades
    return audio

# Define el modelo VAE
class VAE(tf.keras.Model):
    def _init_(self, original_dim, latent_dim):
        super(VAE, self)._init_()
        self.encoder = models.Sequential([
            layers.InputLayer(input_shape=(original_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(latent_dim + latent_dim)  # Media y log var
        ])
        self.decoder = models.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(original_dim, activation='sigmoid')
        ])

    def encode(self, x):
        mu_logvar = self.encoder(x)
        mu, logvar = tf.split(mu_logvar, num_or_size_splits=2, axis=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps

    def call(self, inputs):
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z)

# Entrenamiento del VAE
def train_vae(data, latent_dim, epochs=50, batch_size=32):
    vae = VAE(original_dim=data.shape[1], latent_dim=latent_dim)
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            mu, logvar = vae.encode(x)
            z = vae.reparameterize(mu, logvar)
            reconstructed = vae.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, reconstructed))
            kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
            loss = reconstruction_loss + kl_loss
        gradients = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        return loss

    for epoch in range(epochs):
        for i in range(0, len(data), batch_size):
            x_batch = data[i:i + batch_size]
            loss = train_step(x_batch)
        print(f'Epoch: {epoch + 1}, Loss: {loss.numpy()}')

# Uso
# data = [load_audio(file) for file in your_audio_files]
# train_vae(np.array(data), latent_dim=16)