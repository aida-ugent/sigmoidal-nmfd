import librosa
import librosa.display
import matplotlib.pyplot as plt
import nmfd.init as init
import os


def load_and_plot_file(f):

    y, sr, S_stft = init.load_audio(f)

    S_envelope = init.stft_to_envelope_matrix(S_stft, sr)
    S = S_envelope
    S = init.spectrogram_to_db(S)

    plt.figure()
    librosa.display.specshow(S, )
    plt.show()


if __name__ == '__main__':

    files = [
        '/home/len/Documents/mir/_datasets/ENST-drums-public/drummer_1/audio/wet_mix/044_phrase_rock_simple_fast_rods.wav',
        '/home/len/Documents/mir/_datasets/ENST-drums-public/drummer_2/audio/wet_mix/044_phrase_rock_simple_slow_sticks.wav',
        '/home/len/Documents/mir/_datasets/ENST-drums-public/drummer_2/audio/wet_mix/046_phrase_rock_simple_fast_sticks.wav',
    ]
    for file_uncropped in files:
        p = os.path.split(file_uncropped)
        p0 = os.path.split(p[0])
        file_cropped = os.path.join(p0[0], 'wet_mix', 'cropped', p[1])
        file_uncropped_dry = os.path.join(p0[0], 'dry_mix', p[1])
        file_uncropped_wet = os.path.join(p0[0], 'wet_mix', p[1])
        load_and_plot_file(file_uncropped_dry)
        load_and_plot_file(file_uncropped_wet)
        load_and_plot_file(file_cropped)
