import matplotlib.pyplot as plt
import pandas as pd
import os
def feature_extraction(audio_path):
    import librosa
    import numpy as np
    print(f'{audio_path}')
    # Load the audio file
    y, sr = librosa.load(audio_path)
    # Extract MFCC features
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40),axis=1)
    my_dict = dict(enumerate(mfccs))

    # Extract pitch
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    # Select the pitch with the highest magnitude at each frame
    pitch = [pitches[:, t][np.argmax(magnitudes[:, t])] for t in range(pitches.shape[1])]

    my_dict['Mean Pitch'] = np.round(np.mean(pitch),2)
    my_dict['Varience of Pitch'] = np.round(np.var(pitch),2)  

    # Calculate short-time Intensity/ Energy
    frame_length = 2048
    hop_length = 512
    energy = np.array([
        sum(abs(y[i:i+frame_length]**2))
        for i in range(0, len(y), hop_length)
    ])
  
    my_dict['Mean Energy'] = np.mean(energy)
    my_dict['Varience of Energy'] = np.var(energy)

    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_energy = np.sum(harmonic**2)
    noise_energy = np.sum(percussive**2)

    # Calculate HNR
    hnr = 10 * np.log10(harmonic_energy / noise_energy)
    my_dict['HNR'] = hnr

    # Calculate zero-crossing rate / speech rate
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    my_dict['ZCR'] = np.mean(zcr)

    rounded_dict = {key: round(value, 2) for key, value in my_dict.items()}
    return rounded_dict
    #print(my_dict)

folder = 'C:/Users/Dell/Downloads/Speech Recognition/Dataset/'
feature = []
i=0
for folder_name in os.listdir(folder):
    directory = folder+folder_name+'/'
    for audio in os.listdir(directory):
        audio_path = directory+audio
        z = feature_extraction(audio_path)
        class_value = i%7
        z['Type'] = class_value  # class 0 = Angry   class 1 = Disgust   class 2 = Fear   class 3 = Happy
        feature.append(z)        # class 4 = Neural  class 5 = Pleasant Surprise   class 6 = Sad
    i = i+1
data = pd.DataFrame(feature)
print(data)
data.to_excel('40_mfcc.xlsx', index=False)