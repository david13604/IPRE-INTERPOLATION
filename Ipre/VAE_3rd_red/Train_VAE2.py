from keras.src.datasets import mnist
import keras

from Esqueleto_VAE2 import Sampling, Encoder_Decoder, VAE
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 20

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()    

    x_train = x_train.astype("float32")/255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32")/255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test

def train(x_train, learning_rate, batch_size, epochs):
    autoencoder= Encoder_Decoder(
        input_shape= (28,28,1),
        conv_filters= (32, 64, 64, 64),
        conv_kernells= (3, 3, 3, 3),
        conv_strides= (1, 2, 2, 1),
        latent_space_dim= 2
    ) 

    vae = VAE(autoencoder)
    vae.summary()
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate= learning_rate))
    vae.fit(x_train, epochs=epochs, batch_size=batch_size)

    return vae

if __name__ == "__main__":
    autoencoder= Encoder_Decoder(
        input_shape= (28,28,1),
        conv_filters= (32, 64, 64, 64),
        conv_kernells= (3, 3, 3, 3),
        conv_strides= (1, 2, 2, 1),
        latent_space_dim= 2
    ) 
    x_train, _, _, _ = load_mnist() #datos de tf listos
    #solo 10000 muestras para que no demore tanto
    variatonal_AE  = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS) 
    variatonal_AE.save("primera_VAE") #hay que implementar el save y load

    variatonal_AE2 = VAE.load("primera_VAE")
    variatonal_AE2.summary() 