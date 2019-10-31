import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

filename = "./sav.wav"

y, sr = librosa.load(filename)
spectrogram = librosa.feature.melspectrogram(y=y,sr=sr)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(spectrogram,
                                             ref=np.max),
                         y_axis='mel', fmax=8000,
                         x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.savefig('spect.png')
#plt.show()