import sys
import os
import librosa
import librosa.display
import random
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

pathMpowerEnfermos = "./audios/audios-mpower/true/"
pathMPowerSanos = "./audios/audios-mpower/false/"

#Rutas donde se encuentran guardados los archivos de audios
pathAudiosEnfermos = "./audios/audios-unlam/audios-letra-A/enfermos/"
pathAudiosSanos = "./audios/audios-unlam/audios-letra-A/sanos/"

#Rutas donde serán guardados los espectrogramas de entrenamiento
pathEntrenamientoEnfermos = "./images-unlam/entrenamiento/enfermos"
pathEntrenamientoSanos = "./images-unlam/entrenamiento/sanos"

#Rutas donde serán guardados los espectrogramas de validación
pathValidacionEnfermos = "./images-unlam/validacion/enfermos"
pathValidacionSanos = "./images-unlam/validacion/sanos"

#Ruta donde serán guardadas las imágenes de prueba
pathTestImagesEnfermos = "./images-unlam/test-images/enfermos"
pathTestImagesSanos = "./images-unlam/test-images/sanos"

#Porcentaje asignado a entrenamiento y validación
porcentajeEntrenamiento = 70
porcentajeValidacion = 30

def main():
    #Crea los directorios especificados (en caso de que no existan)
    crearDirectorios()

    recorrerAudios()

    print("Todos los espectrogramas fueron creados")

def porcentaje(num, porciento):
    return int((num*porciento)/100)

def crearDirectorios():
    if not os.path.exists(pathEntrenamientoEnfermos):
	    os.makedirs(pathEntrenamientoEnfermos)

    if not os.path.exists(pathEntrenamientoSanos):
	    os.makedirs(pathEntrenamientoSanos)

    if not os.path.exists(pathValidacionEnfermos):
	    os.makedirs(pathValidacionEnfermos)

    if not os.path.exists(pathValidacionSanos):
	    os.makedirs(pathValidacionSanos)

    if not os.path.exists(pathTestImagesEnfermos):
	    os.makedirs(pathTestImagesEnfermos)

    if not os.path.exists(pathTestImagesSanos):
	    os.makedirs(pathTestImagesSanos)

def recorrerAudios():
    listaEnfermos = os.listdir(pathAudiosEnfermos)
    listaSanos = os.listdir(pathAudiosSanos)

    #Resultados de los porcentajes realizados sobre las listas
    cantEntrenamientoEnfermos = porcentaje(len(listaEnfermos), porcentajeEntrenamiento)
    cantValidacionEnfermos = porcentaje(len(listaEnfermos), porcentajeValidacion)
    cantEntrenamientoSanos = porcentaje(len(listaSanos), porcentajeEntrenamiento)
    cantValidacionSanos = porcentaje(len(listaSanos), porcentajeValidacion)

    # Cada foreach crea espectrogramas con ejes dentro de las carpetas indicadas, de acuerdo a los porcentajes que se hayan definido

    # ENTRENAMIENTO
    for audio in listaEnfermos[0:cantEntrenamientoEnfermos]:
        spectrogram(pathAudiosEnfermos, audio, pathEntrenamientoEnfermos, True)

    # VALIDACION
    for audio in listaEnfermos[cantEntrenamientoEnfermos:cantEntrenamientoEnfermos+cantValidacionEnfermos]:
        spectrogram(pathAudiosEnfermos, audio, pathValidacionEnfermos, False)

    # TEST IMAGES
    for audio in listaEnfermos[cantEntrenamientoEnfermos+cantValidacionEnfermos:len(listaEnfermos)]:
        spectrogram(pathAudiosEnfermos, audio, pathTestImagesEnfermos, False)

    # ENTRENAMIENTO
    for audio in listaSanos[0:cantEntrenamientoSanos]:
        spectrogram(pathAudiosSanos, audio, pathEntrenamientoSanos, True)

    # VALIDACION
    for audio in listaSanos[cantEntrenamientoSanos:cantEntrenamientoSanos+cantValidacionSanos]:
        spectrogram(pathAudiosSanos, audio, pathValidacionSanos, False)

    # TEST IMAGES
    for audio in listaSanos[cantEntrenamientoSanos+cantValidacionSanos:len(listaSanos)]:
        spectrogram(pathAudiosSanos, audio, pathTestImagesSanos, False)

def removerEjes():
    # Removemos los bordes del espectrograma y ajustamos su tamaño (figsize)
    fig, ax = plt.subplots(1, figsize=(6,4))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    # Removemos los ejes del espectrograma
    ax.axis('off')

def guardarGrafico(carpetaDestino, file):
    # Guardamos la imagen en el directorio
    plt.savefig(carpetaDestino + '/' + file[:-4] + '.png')
    plt.close()

def loadAudioFile(path, file):
    # input_length = 16000
    data, sr = librosa.load(path + file)
    # Esto es para agregar padding al audio, en caso de que sea más corto de lo que queremos
    # if len(data)>input_length:
    #     data = data[:input_length]
    # else:
    #     data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data, sr

def randomSignalRoll(data):
    # Shifting the sound
    return np.roll(data, np.random.randint(0, len(data)))

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def spectrogram(path, file, carpetaDestino, preprocess):
    data, sr = loadAudioFile(path, file)
    
    if (preprocess):
        data = randomSignalRoll(data)

        fs = sr # Sample rate
        lowcut = np.random.randint(0, 4000)
        highcut = np.random.randint(4000, 8000)
        y = butter_bandpass_filter(data, lowcut, highcut, fs, order=6)
    else:
        y = data

    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512)
    removerEjes()

    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), fmax=8000, cmap='gray_r')
    guardarGrafico(carpetaDestino, file)


main()