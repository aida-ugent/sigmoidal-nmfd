'''
    This script calculates the mean spectrogram of the samples in the provided sample files, and then saves the result
    in a .npy array.

    Copyright 2020 Len Vande Veire

    This file is part of the code for Sigmoidal NMFD.
    Sigmoidal NMFD is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
    License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
    version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along with this program.
    If not, see <https://www.gnu.org/licenses/>.
'''

import argparse
import csv
import librosa
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate the mean spectrum of the provided samples'
                                                 'and save the result in a .npy array.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sample-files', nargs='+', type=str,
                       help='Input sample files')
    group.add_argument('--samples-list-file', type=str,
                       help='Text file containing a list of sample files.')
    parser.add_argument('--output-file', type=str, required=False,
                        help='Output file')
    parser.add_argument('--plot', action='store_true', required=False,
                        help='Plot the resulting averaged spectrum.')
    parser.add_argument('--spectrum-type', type=str, required=False, default='stft',
                        help="Spectrum type ('stft', 'cqt').")
    parser.add_argument('--template-width', type=int, required=False, default=50,
                        help="Template width in CQT time bins.")
    parser.add_argument('--average-mode', type=str, required=False, default='geometric',
                        help="Average spectrograms of samples either using arithmetic mean or geometric mean.")

    args = parser.parse_args()

    if args.samples_list_file:
        with open(args.samples_list_file) as samples_file:
            reader = csv.reader(samples_file)
            sample_files = [l[0] for l in reader]
    else:
        sample_files = args.sample_files

    HOP_LENGTH = 512

    if args.spectrum_type == 'stft':
        spectrogram_function = librosa.stft
        spectrogram_function_kwargs = {
            'n_fft' : 2048,
            'hop_length': HOP_LENGTH,
        }
    elif args.spectrum_type == 'cqt':
        spectrogram_function = librosa.cqt
        spectrogram_function_kwargs = {
            'sr': 44100,
            'hop_length': HOP_LENGTH,
            'n_bins': 7 * 12 * 1,
            'filter_scale' : 0.5,
        }
    else:
        raise Exception(f'Spectrum type not understood: {args.spectrum_type}')

    if args.plot:
        plt.figure()

    spectrograms = []  # contains the average spectrum for each song individually
    for f in sample_files:
        y, sr = librosa.load(f, sr=44100)

        # Pad the audio with a couple of hop lengths before and after the audio.
        # Before: 2 hops
        # After: template_width hops. This ensures that each sample is at least the desired length, and we only
        # need to crop to get the template.
        y = np.pad(y, (512, 512 * args.template_width))

        # Calculate the spectrogram
        S = spectrogram_function(y, **spectrogram_function_kwargs)
        S = np.abs(S)

        # Trim it to the correct size
        S = S[:,:args.template_width]

        if args.plot:
            plt.plot(S.mean(axis=1))

        spectrograms.append(S)

    if args.plot:
        plt.show()

    if args.average_mode == 'geometric':
        averaged_template = np.exp(np.mean(np.log(np.array(spectrograms)+1e-18), axis=0))
    elif args.average_mode == 'arithmetic':
        averaged_template = np.mean(np.array(spectrograms), axis=0)
    else:
        raise Exception("Spectrogram averaging mode not understood.")
    averaged_template /= np.mean(spectrograms)

    if args.plot:
        plt.figure()
        plt.imshow(averaged_template, origin='lower', aspect='auto')
        plt.show()

        plt.plot(averaged_template.mean(axis=1))
        plt.show()

    if args.output_file:
        np.save(args.output_file, averaged_template)
