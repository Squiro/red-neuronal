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
porcentajeEntrenamiento = 70
porcentajeValidacion = 20

#Offset y duracion que usamos para los audios de mPower
#Si se van a crear esos espectrogramas, deberiamos enviar dichos parametros a las funciones
# off = 2.5
# dur = 2.0

def main():
    #Crea los directorios especificados (en caso de que no existan)
    crearDirectorios()

    print("Elija el tipo de gráfico a crear")
    print("1- Espectrograma con ejes")
    print("2- Espectrograma sin ejes")
    print("3- MFCC (Coeficientes Cepstrales) con ejes") 
    print("4- MFCC (Coeficientes Cepstrales) sin ejes") 
    
    num = int(input("Selecciona: "))

    if (num == 1): 
        recorrerAudios(crearEspectrogramaConEjes)
    elif (num == 2):
        recorrerAudios(crearEspectrogramaSinEjes)
    elif (num == 3):
        recorrerAudios(crearMFCCConEjes)
    elif (num == 4):
        recorrerAudios(crearMFCCSinEjes)  

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
        metodoAEjecutar(pathAudiosEnfermos, audio, pathEntrenamientoEnfermos)
    for audio in listaEnfermos[cantEntrenamientoEnfermos:cantEntrenamientoEnfermos+cantValidacionEnfermos]:
        metodoAEjecutar(pathAudiosEnfermos, audio, pathValidacionEnfermos)
    for audio in listaEnfermos[cantEntrenamientoEnfermos+cantValidacionEnfermos:len(listaEnfermos)]:
        metodoAEjecutar(pathAudiosEnfermos, audio, pathTestImagesEnfermos)

    for audio in listaSanos[0:cantEntrenamientoSanos]:
        metodoAEjecutar(pathAudiosSanos, audio, pathEntrenamientoSanos)
    for audio in listaSanos[cantEntrenamientoSanos:cantEntrenamientoSanos+cantValidacionSanos]:
        metodoAEjecutar(pathAudiosSanos, audio, pathValidacionSanos)
    for audio in listaSanos[cantEntrenamientoSanos+cantValidacionSanos:len(listaSanos)]:
        metodoAEjecutar(pathAudiosSanos, audio, pathTestImagesSanos)   
    
def crearMFCCSinEjes(path, file, carpetaDestino, off=0.0, dur=None):
    try: 
        y, sr = librosa.load(path + file, offset=off, duration=dur)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)             
        removerEjes()
        librosa.display.specshow(mfcc)       
        guardarGrafico(carpetaDestino, file)
    except Exception as e:
        print(e)

def crearMFCCConEjes(path, file, carpetaDestino, off=0.0, dur=None):
    try: 
        y, sr = librosa.load(path + file, offset=off, duration=dur)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)             
        plt.figure(figsize=(6,4))
        librosa.display.specshow(mfcc, x_axis='time')     
        plt.colorbar()  
        guardarGrafico(carpetaDestino, file)
    except Exception as e:
        print(e)

def crearEspectrogramaConEjes(path, file, carpetaDestino, off=0.0, dur=None):
    try: 
        # offset: cuantos segundos nos desplazamos desde el archivo original (float)
        # duration: cuantos segundos de audio leemos (float)
        # Podemos utilizar el offset y la duracion para evitar los problemas que hayan en el audio grabado.
        y, sr = librosa.load(path + file, offset=off, duration=dur)

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
        # para análisis de voz, se recomienda utilizar n_ftt = 512 (lo que da una window lenght de 23ms, similar a la que usaron en el paper)
        # win_length es para dividir cada trama del audio en ventanas, si no se indica este parámetro por defecto es igual a n_ftt
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512)
        plt.figure(figsize=(6,4))
        
        # Power_to_db convierte un espectrograma a unidades de decibeles
        # fmax es un parámetro para definir cuál es la frecuencia máxima
        #librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), fmax=8000)
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')        
        guardarGrafico(carpetaDestino, file)
    except Exception as e: 
        print(e)

def crearEspectrogramaSinEjes(path, file, carpetaDestino, off=0.0, dur=None):
    try: 
        y, sr = librosa.load(path + file, offset=off, duration=dur)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512)
        removerEjes()
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), fmax=8000)        
        guardarGrafico(carpetaDestino, file)
    except Exception as e: 
        print(e)

#def crearEspectrogramaDelta():


#def cargarArchivo(path, file, off, dur):
#    return librosa.load(path + file, offset=off, duration=dur)

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

main()