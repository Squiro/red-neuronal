# Necesitamos tener instalados los packages: librosa, matplotlib
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann

CURR_LETTER = "I"
pathAudios = "../audios/unlam/2020/procesados/" + CURR_LETTER

#Rutas base donde serán guardados los espectrogramas creados
baseTrainingPath = "../spectrograms/"+CURR_LETTER+"/training"
baseValidationPath = "../spectrograms/"+CURR_LETTER+"/validation"
baseTestPath = "../spectrograms/"+CURR_LETTER+"/test"

# Rutas base
pathEnfermos = "/enfermos/"
pathSanos = "/sanos/"

#Porcentaje asignado a entrenamiento y validación
porcentajeEntrenamiento = 60
porcentajeValidacion = 20

hop_length = 512

# List of color maps for the spectogram creation
C_MAPS = [ "magma", "viridis", "gray_r" ]

def main():
	#Crea los directorios especificados (en caso de que no existan)
    crearDirectorios()

    print("Elija el tipo de gráfico a crear")
    print("1- Melspectrogram")
    print("2- MFCC (Coeficientes Cepstrales)") 
    print("3- Log spectrogram")
    
    num = int(input("Selecciona: "))

    if (num == 1): 
        recorrerAudios(melSpectrogram)
    elif (num == 2):
        recorrerAudios(mfcc)  
    elif (num == 3):
        recorrerAudios(logSpectrogram)

    print("Todos los espectrogramas fueron creados")

def porcentaje(num, porciento):   
    return int((num*porciento)/100) 

def crearDirectorios():    
    for cmap in C_MAPS:
        cmapPath = "/"+cmap+"/"
        if not os.path.exists(baseTrainingPath+pathEnfermos+cmapPath):
            os.makedirs(baseTrainingPath+pathEnfermos+cmapPath)
        if not os.path.exists(baseValidationPath+pathEnfermos+cmapPath):
            os.makedirs(baseValidationPath+pathEnfermos+cmapPath)    
        if not os.path.exists(baseTestPath+pathEnfermos+cmapPath):
            os.makedirs(baseTestPath+pathEnfermos+cmapPath)    

        if not os.path.exists(baseTrainingPath+pathSanos+cmapPath):
            os.makedirs(baseTrainingPath+pathSanos+cmapPath)
        if not os.path.exists(baseValidationPath+pathSanos+cmapPath):
            os.makedirs(baseValidationPath+pathSanos+cmapPath)    
        if not os.path.exists(baseTestPath+pathSanos+cmapPath):
            os.makedirs(baseTestPath+pathSanos+cmapPath)    

def recorrerAudios(method):
    for cmap in C_MAPS:
        pathAudiosEnfermos = pathAudios+pathEnfermos
        pathAudiosSanos = pathAudios+pathSanos

        # File list
        listaEnfermos = os.listdir(pathAudiosEnfermos)
        listaSanos = os.listdir(pathAudiosSanos)

        #Resultados de los porcentajes realizados sobre las listas
        cantEntrenamientoEnfermos = porcentaje(len(listaEnfermos), porcentajeEntrenamiento)
        cantValidacionEnfermos = porcentaje(len(listaEnfermos), porcentajeValidacion)
        cantEntrenamientoSanos = porcentaje(len(listaSanos), porcentajeEntrenamiento)
        cantValidacionSanos = porcentaje(len(listaSanos), porcentajeValidacion)

        generate(listaEnfermos, method, pathAudiosEnfermos, pathEnfermos+cmap, cantEntrenamientoEnfermos, cantValidacionEnfermos, cmap)
        generate(listaSanos, method, pathAudiosSanos, pathSanos+cmap, cantEntrenamientoSanos, cantValidacionSanos, cmap)   

def generate(dirList, method, audio_path, save_path, trainingAmount, validationAmount, cmap):
    # Cada foreach crea espectrogramas con ejes dentro de las carpetas indicadas, de acuerdo a los porcentajes que se hayan definido
    for audio in dirList[0:trainingAmount]:
        method(audio_path, audio, baseTrainingPath+save_path, cmap=cmap)
    for audio in dirList[trainingAmount:trainingAmount+validationAmount]:
        method(audio_path, audio, baseValidationPath+save_path, cmap=cmap)
    for audio in dirList[trainingAmount+validationAmount:len(dirList)]:
        method(audio_path, audio, baseTestPath+save_path, cmap=cmap) 

def mfcc(path, file, save_path, off=0.0, dur=None, cmap="magma"):
    try: 
        y, sr = librosa.load(path + file, offset=off, duration=dur)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)             
        removeAxis()
        librosa.display.specshow(mfcc, cmap=cmap)       
        saveSpec(save_path, file)
    except Exception as e:
        print(e)

def melSpectrogram(path, file, save_path, off=0.0, dur=None, cmap="magma"):
    try: 
        # offset: cuantos segundos nos desplazamos desde el archivo original (float)
        # duration: cuantos segundos de audio leemos (float)
        # Podemos utilizar el offset y la duracion para evitar los problemas que hayan en el audio grabado.
        y, sr = librosa.load(path + file, offset=off, duration=dur)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=hop_length*2, hop_length=hop_length, window=hann)  
        # Librosa utiliza una STFT para realizar los espectrogramas. La STFT utiliza una "ventana de tiempo".
        # 
        # Las ventanas de tiempo son muy importantes e influyen en el resultado del espectrograma, debido a que uno de los problemas que tiene STFT es que, dependiendo del tamaño de la ventana,
        # puede perder resolución de frecuencia, o resolución de tiempo.
        #
        # La "resolución de frecuencia" es lo que nos permite identificar con claridad el valor de la frecuencia en el espectrograma. Si perdemos resolución de frecuencia en un espectrograma,
        # nos va a costar distinguir el valor de la frecuencia.
        #
        # La "resolución de tiempo" nos indica el tiempo en que las frecuencias cambian. Si perdemos resolución de frecuencia, en el espectrograma no vamos a notar claramente cuándo
        # cambian esas frecuencias.
        # 
        # La regla es la siguiente: si disminuimos el tamaño de la ventana, aumenta la resolución de tiempo. Si aumentamos el tamaño de la ventana, aumenta la resolución
        # de la frecuencia.
        #
        # Parámetros:
        # n_ftt se usa para establecer el tamaño de la ventana en la STFT
        # para análisis de voz, se recomienda utilizar n_ftt = 512 (lo que da una window lenght de 23ms, similar a la que es usada en varios papers)
        # win_length es para dividir cada trama del audio en ventanas, si no se indica este parámetro por defecto es igual a n_ftt
        removeAxis()      
        # Power_to_db convierte un espectrograma a unidades de decibeles
        # fmax es un parámetro para definir cuál es la frecuencia máxima
        #librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), fmax=8000)
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), fmax=8000, cmap=cmap)
        saveSpec(save_path, file)
    except Exception as e: 
        print(e)

def logSpectrogram(path, file, save_path, off=0.0, dur=None, cmap="magma"):
    try:
        y, sr = librosa.load(path + file, offset=off, duration=dur)
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)), ref=np.max)
        removeAxis()
        librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, cmap=cmap)
        saveSpec(save_path, file)
    except Exception as e:
        print(e)

def removeAxis():
    # Removemos los bordes del espectrograma y ajustamos su tamaño (figsize)
    fig, ax = plt.subplots(1, figsize=(6,4))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    # Removemos los ejes del espectrograma
    ax.axis('off')

def saveSpec(save_path, file): 
    # Guardamos la imagen en el directorio
    plt.savefig(save_path + '/' + file[:-4] + '.png')
    plt.close()

main()