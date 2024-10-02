import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.io.wavfile import write

# Paso 1: Leer la respuesta al impulso desde el archivo 'data.txt'
# Suponiendo que los datos están en formato de una columna (una muestra por línea)
impulse_response = np.loadtxt('respuesta_al_impulso.txt')

# Parámetros de la sinusoide
f = 440  # Frecuencia de la sinusoide (La = 440 Hz)
fs = 44100  # Frecuencia de muestreo (44.1 kHz)
duration = 5  # Duración de la sinusoide en segundos

# Paso 2: Generar la sinusoide (La 440 Hz) que dura 5 segundos
t = np.linspace(0, duration, fs * duration)  # Vector de tiempo
sinusoid = np.sin(2 * np.pi * f * t)

# Paso 3: Realizar la convolución entre la sinusoide y la respuesta al impulso
output_signal = convolve(sinusoid, impulse_response, mode='full')

# Paso 4: Normalizar la señal de salida
output_signal = output_signal / np.max(np.abs(output_signal))

# Convertir la señal a formato int16 para guardarla como archivo .wav
output_signal_int16 = np.int16(output_signal * 32767)

# Guardar la señal convolucionada en un archivo .wav
write('output_convoluted.wav', fs, output_signal_int16)

# Graficar la sinusoide original y la salida convolucionada (solo los primeros 1000 puntos para visualización)
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t[:1000], sinusoid[:1000])
plt.title('Entrada: Sinusoide (La 440 Hz)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')

plt.subplot(2, 1, 2)
plt.plot(output_signal[:1000])
plt.title('Salida: Señal Convolucionada')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')

plt.tight_layout()
plt.show()

print("El archivo de salida se ha guardado como 'output_convoluted.wav'.")
