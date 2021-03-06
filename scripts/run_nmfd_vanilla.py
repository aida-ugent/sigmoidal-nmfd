"""
    Copyright 2020 Len Vande Veire.

    This file is part of the code for Sigmoidal NMFD.
    Sigmoidal NMFD is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
    License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
    version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along with this program.
    If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import numpy as np

import nmfd.vanilla as nmfdvanilla
import nmfd.display as display
import nmfd.init as init


def run_nmfd_vanilla(source_file, n_components, out_dir=None, is_plot=True, annotation_file=None,
                     sparsity=0.0, do_warmup=False):
    y, sr, S_stft = init.load_audio(source_file)

    T_max = 50
    n_bins, n_frames = S_stft.shape
    n_template_frames = T_max

    S_envelope = init.stft_to_envelope_matrix(S_stft, sr)
    S = S_envelope
    S = init.spectrogram_to_db(S)

    initH = 1e-3 * (np.random.random((n_components, S_stft.shape[1])))
    initW_stft = init.initialize_templates(n_bins, n_template_frames, n_components, mode='stft')

    initW = initW_stft
    initW = [init.stft_to_envelope_matrix(W, sr) for W in initW]
    initW = [init.spectrogram_to_db(W) for W in initW]

    paramNMFD = dict()
    paramNMFD['numComp'] = n_components
    paramNMFD['numFrames'] = n_frames
    paramNMFD['numIter'] = 240
    paramNMFD['numTemplateFrames'] = n_template_frames
    paramNMFD['initW'] = initW
    paramNMFD['initH'] = initH
    paramNMFD['record_updates'] = True
    paramNMFD['do_warmup'] = do_warmup

    paramNMFD['sparsity'] = sparsity

    S_padded = np.pad(S, ((0, 0), (0, n_template_frames - 1)), constant_values=1e-9, mode='constant')
    result = nmfdvanilla.nmfd_vanilla(S_padded, paramNMFD)

    paramNMFD['S_stft'] = S
    display.display_result(source_file, result, paramNMFD, amplitude_to_db=False, annotation_file=annotation_file,
                                   out_dir=out_dir, is_plot=is_plot)

    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Apply vanilla NMFD to a drum extract.')
    parser.add_argument('filepath', type=str,
                        help='Path to a .wav or .mp3 file of the audio to be decomposed.')
    parser.add_argument('num_components', type=int,
                        help='Number of NMFD components.')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    run_nmfd_vanilla(
        args.filepath,
        args.num_components,
        is_plot=args.plot,
        out_dir='.' if args.save else None,
    )
