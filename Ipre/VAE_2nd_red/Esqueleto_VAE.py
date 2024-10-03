import tensorflow as tf
import keras
from keras import layers, Model, ops
import numpy as np
from keras.src.optimizers import Adam
from keras.src.losses import MeanSquaredError
import os
import pickle
import numpy as np
print(tf.__version__)
#lo hago con red convolucional


class VAE:

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernells,
                 conv_strides,
                 latent_space_dimm) -> None:
        
        '''
        los parametros conv son tipicos de 
        redes convolucionales que tengo en mi
        cuaderno explicadas, las listas de ejemplos
        se ven como capas (layers)
        ''' 
        self.input_shape = input_shape #[28, 28, 1]
        self.conv_filters = conv_filters #[2, ,4, 8]
        self.conv_kernells = conv_kernells #[3, 5, 3]
        self.conv_strides = conv_strides #[1, 2, 2]
        self.latent_space_dim = latent_space_dimm #2
        self.reconstruction_loss_weight = 1000 #por que????

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build() #encargado de construir encoder decoder y model

    def sumary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    #entrenamiento

    @tf.function
    def compile(self, learning_rate = 0.0001):
        optimizer = Adam(learning_rate= learning_rate)
        #mean_sq_err = MeanSquaredError()
        #self.model.compile(optimizer= optimizer, loss= mean_sq_err)
        self.model.compile(optimizer= optimizer, 
                           loss= self.get_loss(self.mu, self.log_variance))

    def tranin(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, 
                       x_train, 
                       batch_size= batch_size,
                       epochs= num_epochs,
                       shuffle= True)

    def save(self, save_folder = "."):
        self._crear_nuevo_sino_existe(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path) #esto lo uso unicamente mas abajo en classmethod

    def reconstruct(self, images):
        latent_representation = self.encoder.predict(images) #metodo de keras
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
        autoencoder = VAE(*parameters)
        
        weights_path = os.path.join(save_folder, ".weights.h5")
        autoencoder.load_weights(weights_path) #interno de keras vamos a ver si funciona

        return autoencoder

    def get_loss(self, mu, log_variance):
        def _calculate_combined_loss(y_target, y_predicted):
            reconstruction_loss = _calculate_resconstruction_loss(y_target, y_predicted)
            Kl_loss = _calculate_Kl_loss(mu, log_variance)

            combined_loss = self.reconstruction_loss_weight*reconstruction_loss + Kl_loss
            return combined_loss

        def _calculate_resconstruction_loss(y_target, y_predicted):
            error = y_target - y_predicted

            reconstruction_loss = tf.reduce_mean(tf.square(error), axis= [1,2,3]) #basicamente el mean ^2 error
            #axis es para calcular el promedio solo de las dimensiones pedidas(batch_size, height, width, channels)
            # dejaria sin promediar channels en este ejemplo. 
            return reconstruction_loss

        def _calculate_Kl_loss(mu, log_variance):
            #la formula la tengo en a tablet
            kl_loss = -0.5*tf.reduce_sum(1+ log_variance - tf.square(mu) - 
                                         tf.exp(log_variance), axis = 1)

            return kl_loss
        
        return _calculate_combined_loss
    
    def _crear_nuevo_sino_existe(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [self.input_shape ,
        self.conv_filters ,
        self.conv_kernells ,
        self.conv_strides ,
        self.latent_space_dim]

        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, ".weights.h5")
        self.model.save_weights(save_path) #esto es propio de keras creo

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

        #nuevamente carga metodos que van a cargar otros mas que defino adelante
    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name= "autoencoder")


    def _build_decoder(self):
        decorder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decorder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decorder_output = self._add_decoder_output(conv_transpose_layers)

        self.decoder = Model(decorder_input, decorder_output, name = "decoder")
    
    def _add_decoder_input(self):
        return layers.Input(shape=(self.latent_space_dim,), name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck) # [1, 2, 4] -> 8
        dense_layer = layers.Dense(num_neurons, name = "decoder_dense")(decoder_input)
        return dense_layer
    
    def _add_reshape_layer(self, dense_layer):
        reshape_layer = layers.Reshape(self._shape_before_bottleneck)(dense_layer)
        return reshape_layer
    
    def _add_conv_transpose_layers(self, x):
        """Añadir bloques convolucionales
        un loop a traves de todos las capas en orden reverso
         y detenerme en la primera capa """
        for layer_index in reversed(range(1, self._num_conv_layers)):
            # [0, 1, 2] -> [2, 1]
            x = self._add_conv_transpose_layer(layer_index, x)
        return x
    def _add_conv_transpose_layer(self, layer_index, x):
        layer_number = self._num_conv_layers - layer_index

        conv_transpose_layer = layers.Conv2DTranspose(
            filters= self.conv_filters[layer_index],
            kernel_size= self.conv_kernells[layer_index],
            strides= self.conv_strides[layer_index],
            padding= "same",
            name = f"decoder_conv_transpose_layer{layer_number}"
        )
        x = conv_transpose_layer(x)
        x = layers.ReLU(name = f"decoder_relu_{layer_number}")(x)
        x = layers.BatchNormalization(name = f"decoder_bn_{layer_number}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = layers.Conv2DTranspose(
            filters= 1,
            kernel_size= self.conv_kernells[0],
            strides= self.conv_strides[0],
            padding= "same",
            name = f"decoder_conv_transpose_layer{self._num_conv_layers}"
        )

        x = conv_transpose_layer(x)
        output_layer = layers.Activation("sigmoid", name = "sigmoid_layer")(x)
        return output_layer

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        ###
        self._model_input = encoder_input #se usa para despues en el autoencoder
        ###
        self.encoder = Model(encoder_input, bottleneck, name = "encoder")
    
    def _add_encoder_input(self):
        return layers.Input(shape = self.input_shape, name= "encoder_input")
    
    def _add_conv_layers(self, encoder_input):
        '''
        este metodo crea todos los bloques
        convolucionales en el encoder
        '''
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        ''' Añade un bloque convolucional a un graph of layers,
        consisting og cond 2d + ReLU + batch normalization'''

        layer_number = layer_index + 1
        conv_layer = layers.Conv2D(
            filters= self.conv_filters[layer_index],
            kernel_size = self.conv_kernells[layer_index],
            strides= self.conv_strides[layer_index],
            padding= "same",
            name = f"encoder_conv_layer_{layer_number}"
        )

        x = conv_layer(x)
        x = layers.ReLU(name = f"encoder_relu{layer_number}")(x)
        x = layers.BatchNormalization(name = f"encoder_bn{layer_number}")(x)
        return x
    
    def _add_bottleneck(self, x):
        ''' Flatten data and add blottleneck (Dense layer)'''
        """ahora con variational se van a tener dos dense layer
         una para la media otra para la varianza """
        self._shape_before_bottleneck = x.shape[1:]
        x = layers.Flatten()(x)

        self.mu = layers.Dense(self.latent_space_dim, name = "mu")(x)
        self.log_variance = layers.Dense(self.latent_space_dim, name = "log_variance")(x)
        
        #lambda function de keras
        def sample_poion_from_normal_distribution(args):
            mu, log_variance = args
            
            batch_size = tf.shape(log_variance)[0]
            epsilon = keras.random.normal(shape = (batch_size, tf.shape(log_variance)[1]),
                                           mean = 0., stddev = 1.) 
            #epsilon es un punto muestreado de la distribucion normal, shape es para que tenga el
            #mismo largo que el tensor mu
            sampled_point = mu + tf.exp(0.5*log_variance)*epsilon #ahora este es un sampled point de z = mu + sigma
            return sampled_point
        x = layers.Lambda(sample_poion_from_normal_distribution, 
                          name = "encoder_output", output_shape= (self.latent_space_dim,))([self.mu, self.log_variance])
        return x
    
if __name__ == "__main__":
    autoencoder = VAE(
        input_shape= (28,28,1),
        conv_filters= (32, 64, 64, 64),
        conv_kernells= (3, 3, 3, 3),
        conv_strides= (1, 2, 2, 1),
        latent_space_dimm= 2
    )

    autoencoder.sumary()
    