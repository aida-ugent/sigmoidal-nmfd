"""
    This is a modified version of the convModel.py file from the original NMFToolbox.

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

import numpy as np

from .shiftOperator import shiftOperator
from .utils import EPS


def convModel(W, H):
    """Convolutive NMF model.

    Parameters
    ----------
    W: array-like
        Tensor holding the spectral templates which can be interpreted as a set of
        spectrogram snippets with dimensions: numBins x numComp x numTemplateFrames

    H: array-like
        Corresponding activations with dimensions: numComponents x numTargetFrames

    Returns
    -------
    lamb: array-like
        Approximated spectrogram matrix

    """
    # the more explicit matrix multiplication will be used
    # numTemplateFrames, numBins, numComp,  = W.shape
    numBins, numComp, numTemplateFrames  = W.shape
    numComp, numFrames = H.shape

    # initialize with zeros
    lamb = np.zeros((numBins, numFrames))

    # this is doing the math as described in [2], eq (4)
    # the alternative conv2() method does not show speed advantages

    for k in range(numTemplateFrames):
        multResult = W[:, :, k] @ shiftOperator(H, k)
        lamb += multResult

    lamb += EPS

    return lamb


def convModel_optimized(W, H):
    """Convolutional NMFD model, optimized for speed wrt the original NMFToolbox implementation.

    Parameters
    ----------
    W: array-like
        Tensor holding the spectral templates which can be interpreted as a set of
        spectrogram snippets with dimensions: numTemplateFrames x numBins x numComponents

    H: array-like
        Corresponding activations with dimensions: numComponents x numTargetFrames

    Returns
    -------
    lamb: array-like
        Approximated spectrogram matrix

    """
    # the more explicit matrix multiplication will be used
    # numTemplateFrames, numBins, numComp,  = W.shape
    numTemplateFrames, numBins, numComp  = W.shape
    numComp, numFrames = H.shape

    # initialize with zeros
    lamb = np.zeros((numBins, numFrames + numTemplateFrames - 1))

    # Convolving each row is faster than with a matrix multiplication (actually vector outer product)
    for k in range(numComp):
        for n in range(numBins):
            lamb[n, :] += np.convolve(H[k, :], W[:,n,k], mode='full') # [:-numTemplateFrames+1]

    # Implementation with matrix multiplication, not as fast:
    # for k in range(numTemplateFrames):
    #     multResult = W[k, :, :] @ shiftOperator(H, k)
    #     lamb += multResult

    lamb += EPS

    return lamb