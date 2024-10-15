import tensorflow as tf
from keras import layers, Model, ops
import numpy as np
from keras.src.optimizers import Adam
from keras.src.losses import MeanSquaredError
import os
import pickle
import keras


# esta clase se va a llamar cuando se construya el espacio latente
class Sampling(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.seed_generators = keras.random.SeedGenerator(1337)

    def call(self, inputs):

        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]

        epsilon = keras.random.normal(shape=(batch, dim), seed = self.seed_generators)

        return z_mean + ops.exp(0.5* z_log_var) * epsilon

class Encoder_Decoder:
    
    def __init__(self,
                 input_shape, 
                 conv_filters,
                 conv_kernells,
                 conv_strides,
                 latent_space_dim)-> None:
        
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernells = conv_kernells
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim #2 para graficar
        
        self.encoder = None
        self.decoder = None

        self.build()


    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def build(self):
        self.build_encoder()
        self.build_decoder()

    def build_encoder(self):
        encoder_input = keras.Input(shape= self.input_shape, name="encoder_input")
        x = encoder_input
        for layer_index in range(len(self.conv_kernells)):
            x = layers.Conv2D(filters=self.conv_filters[layer_index],
                              kernel_size= self.conv_kernells[layer_index],
                              activation= "relu",
                              strides= self.conv_strides[layer_index],
                              padding= "same")(x)
            
        x = layers.Flatten()(x) #1D
        x = layers.Dense(16, activation= "relu")(x) #16 neuronas

        z_mean = layers.Dense(self.latent_space_dim, name = "z_mean")(x)
        z_log_var = layers.Dense(self.latent_space_dim, name = "z_log_var")(x)

        z = Sampling()([z_mean, z_log_var]) #aca llamo para que haga la normal

        self.encoder = Model(encoder_input, [z_mean,z_log_var, z], name= "encoder")


    def build_decoder(self):
        latent_inputs = keras.Input(shape= (self.latent_space_dim,))
        x = layers.Dense(7*7*64, activation= "relu")(latent_inputs) #shape before blottleneck
        x = layers.Reshape((7,7,64))(x) #reconstruyendo
        for layer_index in reversed(range(len(self.conv_filters))):
            x = layers.Conv2DTranspose(filters= self.conv_filters[layer_index],
                                       kernel_size= self.conv_kernells[layer_index],
                                       activation= "relu",
                                       strides= self.conv_strides[layer_index],
                                       padding="same")(x)
            
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation= "sigmoid", padding="same")(x)
        self.decoder = Model(latent_inputs, decoder_outputs, name = "decoder")

class VAE(keras.Model):
    def __init__(self, encoder_decoder: Encoder_Decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder_decoder = encoder_decoder
        self.encoder = encoder_decoder.encoder
        self.decoder = encoder_decoder.decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    

    ##########################################################
    #### guardar parametros ###########################
    ########################################################
    def save(self, save_folder = "."):
        self._crear_nuevo_sino_existe(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        super().load_weights(weights_path) #esto lo uso unicamente mas abajo en classmethod

    def reconstruct(self, images):
        z_mean, z_log_var, latent_representation = self.encoder.predict(images) #metodo de keras
        reconstructed_images = self.decoder.predict(latent_representation)

        """Lo que hacemos aca es con encoder reducir la info a 
        el espacio latente, luego le damos esto como entrada al decoder
        en escencia solo nos interesa reconstructed pero retorno ambas"""
        return reconstructed_images, latent_representation
    
    @classmethod #por que si no estoy usando cls??
    def load(cls, save_folder = "."):
        parameters_path = os.path.join(save_folder, "parameters.pkl") #el que guarde antes
        with open (parameters_path, "rb") as f:
            parameters = pickle.load(f) #investigar

        autoencoder = Encoder_Decoder(*parameters)
        vae = VAE(autoencoder)
        
        weights_path = os.path.join(save_folder, ".weights.h5")
        vae.load_weights(weights_path) #interno de keras vamos a ver si funciona

        return vae

    def _crear_nuevo_sino_existe(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [self.encoder_decoder.input_shape ,
        self.encoder_decoder.conv_filters ,
        self.encoder_decoder.conv_kernells ,
        self.encoder_decoder.conv_strides ,
        self.encoder_decoder.latent_space_dim]

        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, ".weights.h5")
        self.save_weights(save_path) #esto es propio de keras creo
    
if __name__ == '__main__':
    autoencoder = Encoder_Decoder(
        input_shape= (28,28,1),
        conv_filters= (32, 64, 64, 64),
        conv_kernells= (3, 3, 3, 3),
        conv_strides= (1, 2, 2, 1),
        latent_space_dim= 2
    )

    autoencoder.summary()