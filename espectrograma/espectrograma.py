# Necesitamos tener instalados los packages: librosa, matplotlib

import sys
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

pathAudiosTrue = "./audios/true/"
pathAudiosFalse = "./audios/false/"

#filename = "./audios/5388153.wav"
#y, sr = librosa.load(filename)
#spectrogram = librosa.feature.melspectrogram(y=y,sr=sr)#,win_length=500)

#fig, ax = plt.subplots(1, figsize=(1,1))
#fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
#ax.axis('off')
#librosa.display.specshow(librosa.power_to_db(spectrogram,
#                                             ref=np.max),
#                         y_axis='mel', fmax=8000,
#                         x_axis='time')

#librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), fmax=8000)

#plt.colorbar(format='%+2.0f dB')
#plt.title('Espectrograma')
#plt.tight_layout()
#plt.savefig('./images/spect.png')
#plt.show()


def main():
    for audio in os.listdir(pathAudiosFalse):
        crearEspectrograma(pathAudiosFalse, audio, "false")

    for audio in os.listdir(pathAudiosTrue):
        crearEspectrograma(pathAudiosTrue, audio, "true")

def crearEspectrograma(path, file, carpetaDestino):
    # Obtenemos el nombre dl archivo sin extension
    filename = os.path.splitext(os.path.basename(file))[0]

    try: 
        # Usamos magia del package librosa para crear el espectrograma
        # https://librosa.github.io/librosa/generated/librosa.core.load.html
        # offset: cuantos segundos nos desplazamos desde el archivo original (float)
        # duration: cuantos segundos de audio leemos (float)
        y, sr = librosa.load(path + file, offset=2.0, duration=3.0)
        spectrogram = librosa.feature.melspectrogram(y=y,sr=sr,n_fft=2500,win_length=2500)     #,win_length=500)
        # Removemos los bordes del espectrograma y ajustamos su tamaño en pixeles (figsize)
        fig, ax = plt.subplots(1, figsize=(3,3))
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        # Removemos los ejes del espectrograma
        ax.axis('off')
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), fmax=8000)
        # Guardamos la imagen en el directorio
        plt.savefig('./images/' + carpetaDestino + '/' + filename + '.png')
        plt.close()
    except Exception as e: 
        print(e)
main()