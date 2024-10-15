import tensorflow as tf
from keras import layers, models
from keras.src.datasets import mnist
from keras.src.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Cargar y preprocesar el dataset MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Redimensionamos las imágenes para que tengan una dimensión de canal
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Convertimos las etiquetas a una codificación one-hot
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Definición del modelo CNN
model = models.Sequential()

# Primera capa convolucional
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Segunda capa convolucional
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Tercera capa convolucional
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Aplanado
model.add(layers.Flatten())

# Capa densa
model.add(layers.Dense(64, activation='relu'))

# Capa de salida con softmax para clasificación en 10 categorías
model.add(layers.Dense(10, activation='softmax'))

# Compilamos el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Mostramos la estructura del modelo
model.summary()

# Entrenamos el modelo
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

# Evaluamos el modelo con los datos de prueba
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f'\nPrecisión en el conjunto de prueba: {test_acc:.4f}')


test_image = test_images[0]  # Imagen
test_label = test_labels[0]  # Etiqueta correspondiente

# Mostrar la imagen
plt.imshow(test_image.reshape(28, 28), cmap='gray')
plt.title("Imagen seleccionada")
plt.show()

# Agregar una dimensión adicional para que la imagen tenga la forma adecuada (1, 28, 28, 1)
test_image_reshaped = np.expand_dims(test_image, axis=0)

# Realizamos la predicción
predictions = model.predict(test_image_reshaped)

# Extraemos el índice con la mayor probabilidad (el número predicho)
predicted_label = np.argmax(predictions)

print(f"El modelo predice que la imagen es el número: {predicted_label}")