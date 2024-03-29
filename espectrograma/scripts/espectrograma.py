# Necesitamos tener instalados los packages: librosa 0.8.1, matplotlib
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
import random

CURR_LETTER = "A"
pathAudios = "../audios/unlam/2020/procesados/" + CURR_LETTER

basePath = "../spectrograms/"+CURR_LETTER+"/"

#Rutas base donde serán guardados los espectrogramas creados
trainingPath = "/training"
validationPath = "/validation"
testPath = "/test"

# Rutas base
pathEnfermos = "/enfermos/"
pathSanos = "/sanos/"

#Porcentaje asignado a entrenamiento y validación
porcentajeEntrenamiento = 60
porcentajeValidacion = 20

# Default sampling rate de los archivos de audio
sampling_rate = 44100

# Tamaño de los frames utilizados FFT. Valores recomendados para speech: entre 23 y 40 ms
# Si sr = 44100 => 0.023 * 44100 = 1014 (frame size)
# Utilizamos el tamaño en milisegundos en vez de cantidad de muestras  
n_fft = 0.025

# Overlapping de los frames, utilizado para prevenir saltos o discontinuidades. Valores recomendados: 50% (+-10%) del tamaño del frame como overlap
# Si sr = 44100 => 0.015 * 44100 = 661 (hop length)
hop_length = n_fft*(50/100)

# List of color maps for the spectogram creation
C_MAPS = [ "magma", "viridis", "gray_r" ]

def main():
	#Crea los directorios especificados (en caso de que no existan)
    crearDirectorios()
    
    print("Elija el tipo de espectrograma a crear:")
    print("1- Mel spectrogram")
    print("2- MFCC (Coeficientes Cepstrales)") 
    print("3- Log spectrogram")
    print("4- Mel spectrogram first derivative")
    print("5- Mel spectrogram second derivative")
    
    num = int(input("Selecciona: "))

    method = melSpectrogram

    if (num == 1): 
        method = melSpectrogram
    elif (num == 2):
        method = mfcc
    elif (num == 3):
        method = logSpectrogram
    elif (num == 4):
        method = melFirstDerivative
    elif (num == 5):
        method = melSecondDerivative

    print("Elija el tipo de slicing a aplicar:")
    print("1- Ninguno (totalidad del audio)")
    print("2- Último segundo") 
    print("3- Mitad del audio") 

    num = int(input("Selecciona: "))
    global slicingType
    if (num == 1):
        slicingType = None
    elif (num == 2): 
        slicingType = 'lastSecond'
    elif (num == 3):
        slicingType = "middle"

    recorrerAudios(method)   

    print("Todos los espectrogramas fueron creados")

def porcentaje(num, porciento):   
    return int((num*porciento)/100) 

def crearDirectorios():    
    for cmap in C_MAPS:
        if not os.path.exists(basePath+cmap+trainingPath+pathEnfermos):
            os.makedirs(basePath+cmap+trainingPath+pathEnfermos)
        if not os.path.exists(basePath+cmap+validationPath+pathEnfermos):
            os.makedirs(basePath+cmap+validationPath+pathEnfermos)    
        if not os.path.exists(basePath+cmap+testPath+pathEnfermos):
            os.makedirs(basePath+cmap+testPath+pathEnfermos)    

        if not os.path.exists(basePath+cmap+trainingPath+pathSanos):
            os.makedirs(basePath+cmap+trainingPath+pathSanos)
        if not os.path.exists(basePath+cmap+validationPath+pathSanos):
            os.makedirs(basePath+cmap+validationPath+pathSanos)    
        if not os.path.exists(basePath+cmap+testPath+pathSanos):
            os.makedirs(basePath+cmap+testPath+pathSanos)    

def recorrerAudios(method):
    for cmap in C_MAPS:
        pathAudiosEnfermos = pathAudios+pathEnfermos
        pathAudiosSanos = pathAudios+pathSanos

        # File list
        listaEnfermos = os.listdir(pathAudiosEnfermos)
        listaSanos = os.listdir(pathAudiosSanos)

        cantEntrenamientoEnfermos = porcentaje(len(listaEnfermos), porcentajeEntrenamiento)
        cantValidacionEnfermos = porcentaje(len(listaEnfermos), porcentajeValidacion)
        cantEntrenamientoSanos = porcentaje(len(listaSanos), porcentajeEntrenamiento)
        cantValidacionSanos = porcentaje(len(listaSanos), porcentajeValidacion)

        generate(listaEnfermos, method, pathAudiosEnfermos, pathEnfermos, cantEntrenamientoEnfermos, cantValidacionEnfermos, cmap)
        generate(listaSanos, method, pathAudiosSanos, pathSanos, cantEntrenamientoSanos, cantValidacionSanos, cmap)   

def generate(dirList, method, audio_path, save_path, trainingAmount, validationAmount, cmap):
    # Cada foreach crea espectrogramas con ejes dentro de las carpetas indicadas, de acuerdo a los porcentajes que se hayan definido
    for audio in dirList[0:trainingAmount]:
        applySlicing(audio_path, audio, cmap, basePath+cmap+trainingPath+save_path, method)
        #offset, dur = calculateSlice(audio_path, audio)
        #method(audio_path, audio, basePath+cmap+trainingPath+save_path, offset, dur, cmap=cmap)

    for audio in dirList[trainingAmount:trainingAmount+validationAmount]:
        applySlicing(audio_path, audio, cmap, basePath+cmap+validationPath+save_path, method)
        #offset, dur = calculateSlice(audio_path, audio)
        #method(audio_path, audio, basePath+cmap+validationPath+save_path, offset, dur, cmap=cmap)

    for audio in dirList[trainingAmount+validationAmount:len(dirList)]:
        applySlicing(audio_path, audio, cmap, basePath+cmap+testPath+save_path, method)
        #offset, dur = calculateSlice(audio_path, audio)
        #method(audio_path, audio, basePath+cmap+testPath+save_path, offset, dur, cmap=cmap) 

def mfcc(path, audio_file, save_path, off=0.0, dur=None, cmap="magma"):
    try: 
        y, sr = librosa.load(path + audio_file, offset=off, duration=dur, sr=None)
        mfcc = librosa.feature.mfcc(y, sr)             
        removeAxis()
        librosa.display.specshow(data=mfcc, sr=sr, cmap=cmap, y_axis='mel')       
        saveSpec(save_path, audio_file)
    except Exception as e:        
        print(e)


def getMel(path, audio_file, off=0.0, dur=None):
    # Librosa utiliza una STFT para realizar los espectrogramas. La STFT utiliza una ventana de tiempo/frame.
    # 
    # Las ventanas de tiempo son muy importantes e influyen en el resultado del espectrograma, debido a que uno de los problemas que tiene STFT es que, dependiendo del tamaño de la ventana,
    # el espectrograma puede perder resolución de frecuencia, o resolución de tiempo.
    #
    # La "resolución de frecuencia" es lo que nos permite identificar con claridad el valor de la frecuencia en el espectrograma. Si perdemos resolución de frecuencia en un espectrograma,
    # nos va a costar distinguir el valor de la frecuencia.
    #
    # La "resolución de tiempo" nos indica el tiempo en que las frecuencias cambian. Si perdemos resolución de tiempo, en el espectrograma no vamos a notar claramente cuándo
    # cambian las frecuencias.
    # 
    # Si disminuimos el tamaño de la ventana, aumenta la resolución de tiempo. Si aumentamos el tamaño de la ventana, aumenta la resolución de la frecuencia.
    #
    # Parámetros:
    # n_ftt se usa para establecer el tamaño de la ventana (frame) en la STFT
    # para análisis de voz, se recomienda utilizar un tamaño de frame entre 23 y 40ms de duración
    # win_length largo de la ventana de la funcion window que es aplicada, si no se indica este parámetro por defecto es igual a n_ftt

    # sr=None preserva el sampling rate original del archivo, de otra forma librosa realiza un resampling a 22050
    y, sr = librosa.load(path + audio_file, offset=off, duration=dur, sr=None)
    # fmin=100, fmax=6800, n_mels=320
    return librosa.feature.melspectrogram(y, sr, n_fft=int(n_fft*sr), hop_length=int(hop_length*sr), window=hann)  

def saveMel(spectrogram, audio_file, save_path, cmap):
    removeAxis()      
    # Power_to_db convierte un espectrograma de amplitude squared a decibeles
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), sr=sampling_rate, hop_length=hop_length, cmap=cmap, y_axis='mel')
    saveSpec(save_path, audio_file)

# offset: cuantos segundos nos desplazamos desde el archivo original (float)
# duration: cuantos segundos de audio leemos (float)
def melSpectrogram(path, audio_file, save_path, off=0.0, dur=None, cmap="magma"):
    try: 
        spectrogram = getMel(path, audio_file, off, dur)
        saveMel(spectrogram, audio_file, save_path, cmap)
    except Exception as e: 
        print(e)

def melFirstDerivative(path, audio_file, save_path, off=0.0, dur=None, cmap="magma"):
    try:
         
        spectrogram = librosa.feature.delta(getMel(path, audio_file, off, dur))
        saveMel(spectrogram, audio_file, save_path, cmap)
    except Exception as e: 
        print(e)

def melSecondDerivative(path, audio_file, save_path, off=0.0, dur=None, cmap="magma"):
    try:
        spectrogram = getMel(path, audio_file, off, dur)
        spectrogram = librosa.feature.delta(spectrogram)
        spectrogram = librosa.feature.delta(spectrogram, order=2)
        saveMel(spectrogram, audio_file, save_path, cmap)
    except Exception as e: 
        print(e)

def logSpectrogram(path, audio_file, save_path, off=0.0, dur=None, cmap="magma"):
    try:
        y, sr = librosa.load(path + audio_file, offset=off, duration=dur, sr=None)
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        removeAxis()
        librosa.display.specshow(spectrogram, sr=sr, cmap=cmap, y_axis='log')
        saveSpec(save_path, audio_file)
    except Exception as e:
        print(e)

# Returns offset and duration
def calculateSlice(path, audio_file):
    if (slicingType == None):
        return 0.0, None

    totalDur = librosa.get_duration(filename=path+audio_file)
    if (slicingType == 'lastSecond'):
        return totalDur-1, 1

def applySlicing(audio_path, audio, cmap, save_path, method):
    if (slicingType == None):
        offset, dur = 0.0, None
        method(audio_path, audio, save_path, offset, dur, cmap=cmap)
        return

    totalDur = librosa.get_duration(filename=audio_path+audio)
    if (slicingType == 'lastSecond'):
        offset, dur = totalDur-1, 1
        method(audio_path, audio, save_path, offset, dur, cmap=cmap)
        return

    if (slicingType == 'middle'):
        method(audio_path, audio, save_path, 0, int(totalDur/2), cmap=cmap)
        method(audio_path, audio, save_path, int(totalDur/2), None, cmap=cmap)
        return

def removeAxis():
    # Removemos los bordes y ejes del espectrograma y ajustamos su tamaño (figsize)
    #fig = plt.subplots() #plt.subplots(1, figsize=(6,4))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace = 0, wspace = 0)

def saveSpec(save_path, audio_file): 
    n = ""
    if (slicingType == "middle"):
        n = random.randint(0,22)
    # Guardamos la imagen en el directorio
    plt.savefig(save_path + '/' + audio_file[:-4] + str(n) + '.png')
    plt.close()

main()