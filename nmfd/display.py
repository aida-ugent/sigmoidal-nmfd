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
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


def filename_to_compact_string(path_to_file):
    return os.path.splitext(os.path.basename(path_to_file))[0]


def display_result(source_file, result, paramNMFD, annotation_file=None,
                   amplitude_to_db=False, out_dir=None, is_plot=True):

    def specshow(S, **kwargs):
        S_ = np.copy(S)
        if amplitude_to_db:
            S_ = librosa.power_to_db(S_, ref=np.max)
        librosa.display.specshow(S_, **kwargs)

    A = result.get('A', None)
    is_sigmoid = A is not None
    nmfdW = result['W_k']
    nmfdH = result['H']
    tensorW = result['tensorW']
    Lambda = result['S_hat']
    K = tensorW.shape[2]
    T = nmfdH.shape[1]

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    widths = [1, 16]
    heights = ([1.5]    # Spectrogram
               + [1.5]  # and approximation
               + [0.5] * K  # H and W
               )
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights)

    n_plot = 0

    # Plot the spectrogram ...
    ax = fig.add_subplot(spec[0, 1])
    specshow(paramNMFD['S_stft'], ax=ax)
    ax.set_xlim((0, T))
    n_plot += 1
    # ... and the spectrogram approximation
    ax = fig.add_subplot(spec[1, 1])
    specshow(Lambda, ax=ax)
    ax.set_xlim((0, T))
    n_plot += 1

    # Plot the activations H and the templates W
    for k in range(nmfdH.shape[0]):
        ax = fig.add_subplot(spec[n_plot, 0])
        specshow(nmfdW[k], ax=ax)

        ax = fig.add_subplot(spec[n_plot, 1])
        ax.plot(nmfdH[k] + 1e-9, c=f'C{k}', linewidth=3)
        if is_sigmoid:
            ax.hlines(A[k], 0, len(nmfdH[k]), linestyle='--', color=f'C{k}', alpha=0.7)
        if not amplitude_to_db or is_sigmoid:
            ax.set_ylim((-0.1, 1.1))
        ax.set_xlim((0, T))

        n_plot += 1

    if out_dir is not None:
        filename = filename_to_compact_string(source_file)
        plt.savefig(os.path.join(out_dir, f'{filename}.png'))
        print(filename)

        arrays_to_save = {
            'S_stft': paramNMFD['S_stft'],
            'S_out': Lambda,
            'H': nmfdH,
            'W': nmfdW,
            'loss': result['costFunc'],
            'source_file': [source_file, ],
            'annotation_file' : [annotation_file, ],
        }
        np.savez(os.path.join(out_dir, f'{filename}.npz'), **arrays_to_save)
    if is_plot:
        plt.show()
    plt.close()
