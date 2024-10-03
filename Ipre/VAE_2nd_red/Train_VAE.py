from keras.src.datasets import mnist

from keras import backend as K

from Esqueleto_VAE import VAE
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
    variatonal_AE = VAE(
        input_shape= (28,28,1),
        conv_filters= (32, 64, 64, 64),
        conv_kernells= (3, 3, 3, 3),
        conv_strides= (1, 2, 2, 1),
        latent_space_dimm= 2
    ) 
    variatonal_AE.sumary()
    variatonal_AE.compile(learning_rate)
    variatonal_AE.tranin(x_train, batch_size, epochs)
    return variatonal_AE 

if __name__ == "__main__":
    x_train, _, _, _ = load_mnist() #datos de tf listos
    #solo 500 muestras para que no demore tanto
    variatonal_AE  = train(x_train[:10000], LEARNING_RATE, BATCH_SIZE, EPOCHS) 
    variatonal_AE.save("primer_autoencoder") #hay que implementar el save y load
    variatonal_AE2 = VAE.load("primer_autoencoder")
    variatonal_AE2.sumary()