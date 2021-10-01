# Necesitamos tener instalados los packages: librosa, matplotlib
import sys
import os
import librosa
import librosa.display
import numpy as np
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
porcentajeEntrenamiento = 100
porcentajeValidacion = 0

# hop_length
hop_length = 512

# duracion de los audios
DURATION=2

def main():
    #Crea los directorios especificados (en caso de que no existan)
    crearDirectorios()
    
    print("Elija el tipo de gráfico a crear")
    print("1- Mel spectrogram sin ejes")
    print("2- Log spectrogram sin ejes")
    
    num = int(input("Selecciona: "))

    if (num == 1):
        recorrerAudios(crearEspectrogramaMelSinEjes)
    elif (num == 2):
        recorrerAudios(crearEspectrogramaLogSinEjes)

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

def recorrerAudios(metodoAEjecutar):
    listaEnfermos = os.listdir(pathAudiosEnfermos)
    listaSanos = os.listdir(pathAudiosSanos)

    #Resultados de los porcentajes realizados sobre las listas
    cantEntrenamientoEnfermos = porcentaje(len(listaEnfermos), porcentajeEntrenamiento)
    cantValidacionEnfermos = porcentaje(len(listaEnfermos), porcentajeValidacion)
    cantEntrenamientoSanos = porcentaje(len(listaSanos), porcentajeEntrenamiento)
    cantValidacionSanos = porcentaje(len(listaSanos), porcentajeValidacion)

    # Cada foreach crea espectrogramas con ejes dentro de las carpetas indicadas, de acuerdo a los porcentajes que se hayan definido
    for audio in listaEnfermos[0:cantEntrenamientoEnfermos]:
        dividirAudioEnMuestras(pathAudiosEnfermos, audio, metodoAEjecutar, pathEntrenamientoEnfermos)
    for audio in listaEnfermos[cantEntrenamientoEnfermos:cantEntrenamientoEnfermos+cantValidacionEnfermos]:
        dividirAudioEnMuestras(pathAudiosEnfermos, audio, metodoAEjecutar, pathValidacionEnfermos)
    for audio in listaEnfermos[cantEntrenamientoEnfermos+cantValidacionEnfermos:len(listaEnfermos)]:
        dividirAudioEnMuestras(pathAudiosEnfermos, audio, metodoAEjecutar, pathTestImagesEnfermos)

    for audio in listaSanos[0:cantEntrenamientoSanos]:
        dividirAudioEnMuestras(pathAudiosSanos, audio, metodoAEjecutar, pathEntrenamientoSanos)
    for audio in listaSanos[cantEntrenamientoSanos:cantEntrenamientoSanos+cantValidacionSanos]:
        dividirAudioEnMuestras(pathAudiosSanos, audio, metodoAEjecutar, pathValidacionSanos)
    for audio in listaSanos[cantEntrenamientoSanos+cantValidacionSanos:len(listaSanos)]:
        dividirAudioEnMuestras(pathAudiosSanos, audio, metodoAEjecutar, pathTestImagesSanos)

def dividirAudioEnMuestras(path, filename, metodoAEjecutar, carpetaDestino): 
    y, sr = librosa.load(path + filename, duration=DURATION)

    # Cantidad de muestras totales presentes en el audio, segun la duracion especificada
    cantidadMuestras = sr*DURATION
    mitadMuestras = int(cantidadMuestras/2)
    mitadDeMitad = int(mitadMuestras/2)
    newFilename = filename[:-4]
    
    # Chequeamos si la longitud del audio que leimos es de DURATION segundos
    if (len(y) == cantidadMuestras):
        # First slice, e.g: 0 - 22050, asumiendo un audio de 2 segundos (44100 muestras) con un SR de 22050 muestras p/s
        metodoAEjecutar(y[0:mitadMuestras], sr, newFilename + " - 1", carpetaDestino)
        # Second slice, e.g: 22050 - 44100
        metodoAEjecutar(y[mitadMuestras:cantidadMuestras], sr, newFilename + " - 2", carpetaDestino)
        # Third slice, e.g: 11025 - 33075
        metodoAEjecutar(y[mitadMuestras-mitadDeMitad:mitadMuestras+mitadDeMitad], sr, newFilename + " - 3", carpetaDestino)

def crearEspectrogramaMelSinEjes(y, sr, filename, carpetaDestino):
    try:             
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=hop_length*2, hop_length=hop_length)  
        removerEjes()        
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), fmax=8000, cmap='viridis') #cmap='gray_r'
        guardarGrafico(carpetaDestino, filename)
    except Exception as e: 
        print(e)

def crearEspectrogramaLogSinEjes(y, sr, filename, carpetaDestino):
    try:
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)), ref=np.max)
        removerEjes()
        librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, cmap='viridis')
        guardarGrafico(carpetaDestino, filename)
    except Exception as e:
        print(e)

def removerEjes():
    # Removemos los bordes del espectrograma y ajustamos su tamaño (figsize)
    fig, ax = plt.subplots(1, figsize=(6,4))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    # Removemos los ejes del espectrograma
    ax.axis('off')

def guardarGrafico(carpetaDestino, filename): 
    # Guardamos la imagen en el directorio
    plt.savefig(carpetaDestino + '/' + filename + '.png')
    plt.close()

main()