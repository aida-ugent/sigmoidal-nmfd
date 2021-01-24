"""
    This is a modified version of the initTemplates.py file from the original NMFToolbox.

    Copyright 2020 Len Vande Veire

    This file is part of the code for Sigmoidal NMFD.
    Sigmoidal NMFD is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
    License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
    version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along with this program.
    If not, see <https://www.gnu.org/licenses/>.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    The original NMFD toolbox is described in:
    [1]  Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
        Müller
        NMF Toolbox: Music Processing Applications of Nonnegative Matrix
        Factorization
        In Proceedings of the International Conference on Digital Audio Effects
        (DAFx), 2019.

    License:
    This original version of this file is part of 'NMF toolbox'.
    https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/
    'NMF toolbox' is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    the Free Software Foundation, either version 3 of the License, or (at
    your option) any later version.

    'NMF toolbox' is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
    Public License for more details.

    You should have received a copy of the GNU General Public License along
    with 'NMF toolbox'. If not, see http://www.gnu.org/licenses/.
"""

import os
import numpy as np
from .utils import EPS


def load_templates_from_npy_files(paths_to_presets, num_bins, num_template_frames):
    """
    Load the templates for initW from the provided .npy files.

    Parameters
    ----------
    paths_to_presets: iterable of strings, being the paths to the template files that are loaded into slices of initW
        These numpy files should contain spectral templates of shape (num_bins, ?). If the template width does not match
        the intended template width, the spectral templates are either zero-padded on the right, or cropped.
    num_bins: Number of spectral bins along the frequency axis.
    num_template_frames: Width of the resulting initW tensor in time frames.

    Returns
    -------
    initW:
        List with the desired templates
    """

    num_components = len(paths_to_presets)
    initW = []
    for i, instrument_preset_path in enumerate(paths_to_presets):
        instrument_preset = np.load(instrument_preset_path)
        if instrument_preset.shape[0] != num_bins:
            raise Exception(f'Number of bins {num_bins} not equal to number of bins of drum templates {instrument_preset.shape[0]}.')
        if instrument_preset.ndim == 1:
            instrument_preset = instrument_preset.reshape(-1,1)
            instrument_preset = instrument_preset * np.exp(-np.linspace(0,5,num_template_frames)).reshape(1,-1)
        else:
            pad_amount = num_template_frames - instrument_preset.shape[1]
            if pad_amount > 0:
                instrument_preset = np.pad(instrument_preset, pad_width=((0, 0), (0, pad_amount)))
            elif pad_amount < 0:
                instrument_preset = instrument_preset[:, :pad_amount]
        initW.append(instrument_preset)
    # for i in range(num_components - 3):
    #     initW.append(np.random.rand(num_bins, num_template_frames))
    return initW

def initTemplates(parameter, strategy='random'):
    """Implements different initialization strategies for NMF templates. The
    strategies 'random' and 'uniform' are self-explaining. The strategy
    'pitched' uses comb-filter templates as described in [2]. The strategy
    'drums' uses pre-extracted, averaged spectra of desired drum types [3].

    References
    ----------
    [2] Jonathan Driedger, Harald Grohganz, Thomas Prätzlich, Sebastian Ewert
    and Meinard Mueller "Score-informed audio decomposition and applications"
    In Proceedings of the ACM International Conference on Multimedia (ACM-MM)
    Barcelona, Spain, 2013.

    [3] Christian Dittmar and Meinard Müller -- Reverse Engineering the Amen
    Break - Score-informed Separation and Restoration applied to Drum
    Recordings" IEEE/ACM Transactions on Audio, Speech, and Language Processing,
    24(9): 1531-1543, 2016.

    Parameters
    ----------
    parameter: dict
        numComp           Number of NMF components
        numBins           Number of frequency bins
        numTemplateFrames Number of time frames for 2D-templates
        pitches           Optional array of MIDI pitch values
        drumTypes         Optional list of drum type strings

    strategy: str
        String describing the initialization strategy

    Returns
    -------
    initW: array-like
        List with the desired templates
    """
    # check parameters
    parameter = init_parameters(parameter)
    initW = list()

    if strategy == 'random':
        # fix random seed
        np.random.seed(0)

        for k in range(parameter['numComp']):
            initW.append(np.random.rand(parameter['numBins'], parameter['numTemplateFrames']))

    elif strategy == 'uniform':
        for k in range(parameter['numComp']):
            initW.append(np.ones((parameter['numBins'], parameter['numTemplateFrames'])))

    elif strategy == 'drums-custom-fulltemplate':

        if parameter['numComp'] == 3:
            instrument_names = ('kick', 'snare', 'hihat')
        elif parameter['numComp'] == 4:
            instrument_names = ('kick', 'snare', 'hihat', 'crash')
        else:
            instrument_names = ['kick', 'hihat', 'snare',  'crash'] + ['hihat', 'snare']*(parameter['numComp'] - 4)

        # Presets should be full templates instead of averaged spectra
        path_to_presets = parameter['drumsTemplateFolder']
        instrument_preset_files = [os.path.join(path_to_presets, f'{instrument_name}.npy')
                              for instrument_name in instrument_names]
        initW = load_templates_from_npy_files(instrument_preset_files, parameter['numBins'], parameter['numTemplateFrames'])

    else:
        raise ValueError('Invalid strategy.')

    # do final normalization
    for k in range(parameter['numComp']):
        initW[k] /= (EPS + initW[k].sum())

    return initW


def init_parameters(parameter):
    """Auxiliary function to set the parameter dictionary

    Parameters
    ----------
    parameter: dict
        See the above function initTemplates for further information

    Returns
    -------
    parameter: dict
    """
    parameter['pitchTolUp'] = 0.75 if 'pitchTolUp' not in parameter else parameter['pitchTolUp']
    parameter['pitchTolDown'] = 0.75 if 'pitchTolDown' not in parameter else parameter['pitchTolDown']
    parameter['numHarmonics'] = 25 if 'numHarmonics' not in parameter else parameter['numHarmonics']
    parameter['numTemplateFrames'] = 1 if 'numTemplateFrames' not in parameter else parameter['numTemplateFrames']

    return parameter
