"""
    Copyright 2020 Len Vande Veire

    This file is part of the code for Sigmoidal NMFD.
    Sigmoidal NMFD is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
    License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
    version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along with this program.
    If not, see <https://www.gnu.org/licenses/>.
"""

import librosa
from .NMFtoolbox import initTemplates
import numpy as np
import os


def spectrogram_to_db(S):
    eps = 1e-9
    eps_prelog = 1e-7
    S_ = librosa.power_to_db(S + eps_prelog, ref=np.max)
    S_min, S_max = S_.min(), S_.max()
    S_ = ((1-eps)*S_ - S_min + eps*S_max) / (S_max - S_min)
    return S_


def __initialize_templates(drum_template_folder, n_bins, n_template_frames, n_components, amplitude_to_db=False):
    paramTemplates = dict()
    paramTemplates['numComp'] = n_components
    paramTemplates['numBins'] = n_bins
    paramTemplates['numTemplateFrames'] = n_template_frames
    paramTemplates['drumsTemplateFolder'] = drum_template_folder
    initW = initTemplates.initTemplates(paramTemplates, 'drums-custom-fulltemplate')

    if amplitude_to_db:
        initW = [librosa.amplitude_to_db(W) for W in initW]

    initW = [W - W.min() for W in initW]
    initW = [W / W.max() for W in initW]
    initW = [W + 1e-18 for W in initW]

    return initW


def initialize_templates(*args, mode='stft', **kwargs):
    if mode == 'stft':
        return initialize_templates_stft(*args, **kwargs)
    else:
        raise Exception(f'Unknown initialization mode "{mode}"')


def initialize_templates_stft(n_bins, n_template_frames, n_components, amplitude_to_db=False):
    drum_template_folder = os.path.join(os.path.dirname(__file__), '..', 'resources/templates/')
    return __initialize_templates(
        drum_template_folder, n_bins, n_template_frames, n_components, amplitude_to_db=amplitude_to_db)


def load_audio(path_to_file):
    sr = 44100
    y, sr = librosa.load(path_to_file, sr=sr)
    # Pad to avoid edge effects
    y = np.concatenate((np.zeros(1024), y, np.zeros(1024)))
    S_stft = np.abs(librosa.stft(y, hop_length=256, n_fft=2048, win_length=1024)) ** 2
    return y, sr, S_stft


def frequency_to_fft_bin(f, n_fft, sr):
    f_range = librosa.fft_frequencies(sr, n_fft)
    bin_range = range(n_fft // 2 + 1)
    return np.interp(f, f_range, bin_range, )


def envelope_from_stft(S_stft, f_min, f_max, sr):
    n_fft = (S_stft.shape[0] - 1) * 2
    lowest_bin = int(np.round(frequency_to_fft_bin(f_min, n_fft, sr)))
    highest_bin = int(np.round(frequency_to_fft_bin(f_max, n_fft, sr)))
    return np.sum(S_stft[lowest_bin:highest_bin, :], axis=0)


def stft_to_envelope_matrix(S, sr):
    frequencies_mel = librosa.mel_frequencies(n_mels=25 + 1, fmin=0, fmax=11025, htk=False)
    frequencies = [(f0, f1) for f0, f1 in zip(frequencies_mel[:-1], frequencies_mel[1:])]
    envelopes = [envelope_from_stft(S, fmin, fmax, sr) for fmin, fmax in frequencies]
    S_ = np.array(envelopes)
    return S_ / np.max(S_)