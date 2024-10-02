
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import os

path = '/Users/usuario/Desktop/Uni/Semestre_8/Ipre/MIT KEMAR Dataset/elev-10'
lista = os.listdir(path)
# Cargar archivo de audio .wav
sample_rate, data = wavfile.read(path + '/' + lista[0])
#print(lista)


print(data)
print(sample_rate)

#NUT = 1, sr = 1/T
N = len(data)
sr = sample_rate
U = sr/(N)

freqs = np.linspace(0, sr, N)
t = np.linspace(0, (N-1)/sr, N)

data_frecs = fft(data)

plt.figure(figsize = (8, 6))
plt.plot(freqs, data_frecs, 'r')
plt.title('frecuencia')

plt.show()

np.savetxt('respuesta_al_impulso.txt', data)
