# Necesitamos tener instalados los packages: librosa, matplotlib
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

#Rutas donde se encuentran guardados los archivos de audios
pathAudiosEnfermos = "../audios/unlam/audios-recortados/clasificados-renombrados/enfermos/U/"
pathAudiosSanos = "../audios/unlam/audios-recortados/clasificados-renombrados/sanos/U/"

#Rutas base donde serán guardados los espectrogramas creados
baseTrainingPath = "../spectrograms/entrenamiento"
baseValidationPath = "../spectrograms/validation"
baseTestPath = "../spectrograms/test-images"

# Rutas base
pathEnfermos = "/enfermos"
pathSanos = "/sanos"

#Porcentaje asignado a entrenamiento y validación
porcentajeEntrenamiento = 100
porcentajeValidacion = 0

# hop_length
hop_length = 512

# Color map
C_MAP = "magma"

def main():
	#Crea los directorios especificados (en caso de que no existan)
    crearDirectorios()

    print("Elija el color map de los espectrogramas")
    print("1- Default color map (magma)")
    print("2- Gray scale color map")
    print("3- Viridis color map")
    input_cmap = int(input("Selecciona: "))

    if (input_cmap == 1):    	
    	C_MAP = "magma"
    elif (input_cmap == 2):
    	C_MAP = "gray_r"
    elif (input_cmap == 3):
    	C_MAP = "viridis"

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
    if not os.path.exists(baseTrainingPath+pathEnfermos):
	    os.makedirs(baseTrainingPath+pathEnfermos)
        
    if not os.path.exists(baseValidationPath+pathEnfermos):
	    os.makedirs(baseValidationPath+pathEnfermos)

    if not os.path.exists(baseTestPath+pathEnfermos):
	    os.makedirs(baseTestPath+pathEnfermos)   

    if not os.path.exists(baseTrainingPath+pathSanos):
	    os.makedirs(baseTrainingPath+pathSanos)   
    
    if not os.path.exists(baseValidationPath+pathSanos):
        os.makedirs(baseValidationPath+pathSanos)   
    
    if not os.path.exists(baseTestPath+pathSanos):
	    os.makedirs(baseTestPath+pathSanos)     

def recorrerAudios(method):
    listaEnfermos = os.listdir(pathAudiosEnfermos)
    listaSanos = os.listdir(pathAudiosSanos)

    #Resultados de los porcentajes realizados sobre las listas
    cantEntrenamientoEnfermos = porcentaje(len(listaEnfermos), porcentajeEntrenamiento)
    cantValidacionEnfermos = porcentaje(len(listaEnfermos), porcentajeValidacion)
    cantEntrenamientoSanos = porcentaje(len(listaSanos), porcentajeEntrenamiento)
    cantValidacionSanos = porcentaje(len(listaSanos), porcentajeValidacion)

    generate(listaEnfermos, method, pathAudiosEnfermos, '/enfermos', cantEntrenamientoEnfermos, cantValidacionEnfermos)
    generate(listaSanos, method, pathAudiosSanos, '/sanos', cantEntrenamientoSanos, cantValidacionSanos)   

def generate(dirList, method, audio_path, save_path, trainingAmount, validationAmount):
    # Cada foreach crea espectrogramas con ejes dentro de las carpetas indicadas, de acuerdo a los porcentajes que se hayan definido
    for audio in dirList[0:trainingAmount]:
        method(audio_path, audio, baseTrainingPath+save_path)
    for audio in dirList[trainingAmount:trainingAmount+validationAmount]:
        method(audio_path, audio, baseValidationPath+save_path)
    for audio in dirList[trainingAmount+validationAmount:len(dirList)]:
        method(audio_path, audio, baseTestPath+save_path) 

def mfcc(path, file, save_path, off=0.0, dur=None):
    try: 
        y, sr = librosa.load(path + file, offset=off, duration=dur)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)             
        removerEjes()
        librosa.display.specshow(mfcc, cmap=C_MAP)       
        guardarGrafico(save_path, file)
    except Exception as e:
        print(e)

def melSpectrogram(path, file, save_path, off=0.0, dur=None):
    try: 
        # offset: cuantos segundos nos desplazamos desde el archivo original (float)
        # duration: cuantos segundos de audio leemos (float)
        # Podemos utilizar el offset y la duracion para evitar los problemas que hayan en el audio grabado.
        y, sr = librosa.load(path + file, offset=off, duration=dur)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=hop_length*2, hop_length=hop_length)  
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
        removerEjes()      
        # Power_to_db convierte un espectrograma a unidades de decibeles
        # fmax es un parámetro para definir cuál es la frecuencia máxima
        #librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), fmax=8000)
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), fmax=8000, cmap=C_MAP)
        guardarGrafico(save_path, file)
    except Exception as e: 
        print(e)

def logSpectrogram(path, file, save_path, off=0.0, dur=None):
    try:
        y, sr = librosa.load(path + file, offset=off, duration=dur)
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)), ref=np.max)
        removerEjes()
        librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, cmap=C_MAP)
        guardarGrafico(save_path, file)
    except Exception as e:
        print(e)

def removerEjes():
    # Removemos los bordes del espectrograma y ajustamos su tamaño (figsize)
    fig, ax = plt.subplots(1, figsize=(6,4))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    # Removemos los ejes del espectrograma
    ax.axis('off')

def guardarGrafico(save_path, file): 
    # Guardamos la imagen en el directorio
    plt.savefig(save_path + '/' + file[:-4] + '.png')
    plt.close()

main()