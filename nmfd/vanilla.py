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

import numpy as np
from copy import deepcopy

from .NMFtoolbox.convModel import convModel_optimized as convModel
from .NMFtoolbox.shiftOperator import shiftOperator
from .sigmoid import get_mu_k, sigmoid_inv


def update_term_H_KL(X, H, tensorW, weight=None, sparsity=0.0,):

    numerator_list = []
    denominator_list = []

    Lambda = convModel(tensorW, H) + 1e-9
    T = tensorW.shape[0]
    R = H.shape[0]

    X_Lambda_ = X * Lambda ** -1
    ones_ = np.ones((Lambda.shape[0], H.shape[1]))

    if weight is None:
        weight = np.ones((R, 1))

    for tau in range(T):
        # compute update terms for shift tau
        transpW = tensorW[tau, :, :].T
        X_Lambda_shifted = X_Lambda_[:, tau:H.shape[1] + tau]
        numerator = transpW @ X_Lambda_shifted
        denominator = sparsity + transpW @ ones_

        # accumulate update term
        numerator_list.append(weight * numerator)
        denominator_list.append(weight * denominator)

    # Calculate best match for each T, if update > 1
    numerator_sum = np.sum(numerator_list, axis=0)
    denominator_sum = np.sum(denominator_list, axis=0)

    return [numerator_sum], [denominator_sum]


def update_term_W_KL(X, H, tensorW, weight=1):
    update = np.ones_like(tensorW)
    Lambda = convModel(tensorW, H) + 1e-9
    T = tensorW.shape[0]

    X_Lambda_ = X * Lambda ** -1
    X_Lambda_ = X_Lambda_[:, :-T + 1]
    ones_ = np.ones_like(X_Lambda_)

    for tau in range(T):
        transpH = shiftOperator(H, tau).T
        numerator = (X_Lambda_) @ transpH
        denominator = ones_ @ transpH
        update[tau, :, :] = numerator / denominator

    return update


def loss_function(X, H, tensorW, penalty_weight=1.0):
    L_tau, N, K = tensorW.shape

    # Spectrogram loss
    Lambda = convModel(tensorW, H) + 1e-18
    loss_spectrogram = np.sum(X * (np.log(X + 1e-18) - np.log(Lambda)) + Lambda - X)

    # Penalty on G
    # H_logit = sigmoid_inv(H / np.max(H, axis=0))
    # mu_k = get_mu_k(H_logit, alpha=0.5)
    # loss_activations = penalty_weight * np.sum(np.exp(-((H_logit - mu_k) / 2) ** 2))

    return loss_spectrogram # + loss_activations


def nmfd_vanilla(S, parameter=None, paramConstr=None):

    NUM_ITER = parameter['numIter']

    # Initialize shape variables
    N, T_length = S.shape
    L_tau = parameter['numTemplateFrames']
    K = parameter['numComp']

    # Initialize L1 sparsity variable
    sparsity = parameter.get('sparsity', 0.0)

    # Initialize tensorW
    tensorW = np.zeros((L_tau, N, K))
    for k in range(K):
        tensorW[:, :, k] = parameter['initW'][k].T

    # Initialize H
    H = deepcopy(parameter['initH'])

    # Initialize target spectrogram
    V_tmp = S / np.max(S)

    # cost function -- for logging
    costFunc = []
    # updates per iteration -- for logging
    record_updates = parameter.get('record_updates', False)
    updates_H = []

    # Perform a warmup: allow some iterations of unconstrained optimization before applying L1 regularization
    do_warmup = parameter.get('do_warmup', False)
    warmup_len = parameter.get('warmup_len', 30)
    if do_warmup:
        sparsity_cycle = warmup_len * [0.0] + (NUM_ITER - warmup_len) * [sparsity]
    else:
        sparsity_cycle = [sparsity]

    def _normalize_tensorW(tensorW, H):
        tensorW /= np.max(tensorW, axis=(0, 1), keepdims=True)

    for iter_ in range(NUM_ITER):

        _normalize_tensorW(tensorW, H)

        # --------------------------------------------
        # --------- UPDATE H -------------------------
        # --------------------------------------------

        # Update of H wrt first objective (spectrogram)
        multH_numerator, multH_denominator = update_term_H_KL(
            V_tmp, H, tensorW, sparsity=sparsity_cycle[iter_ % len(sparsity_cycle)])

        multH_numerator = np.sum(multH_numerator, axis=0)
        multH_denominator = np.sum(multH_denominator, axis=0)
        multH = multH_numerator / multH_denominator

        if record_updates:
            updates_record = {
                'iter_': iter_,
                'S': np.copy(V_tmp),
                'S_hat': convModel(tensorW, H),
                'H': np.copy(H),
                'W': np.copy(tensorW),
                'multH': multH,
                'multH_numerator': multH_numerator,
                'multH_denominator': multH_denominator,
            }

            updates_H.append(updates_record)

            H *= multH

        # --------------------------------------------
        # --------- UPDATE W -------------------------
        # --------- (Spectrogram templates) ----------
        # --------------------------------------------
        tensorW *= update_term_W_KL(V_tmp, H, tensorW)
        _normalize_tensorW(tensorW, H)

        # store the divergence with respect to the target spectrogram
        loss = loss_function(V_tmp, H, tensorW)
        costFunc.append(loss)

    # Overall approximation
    Lambda_V = convModel(tensorW, H)

    # Per-component approximations
    W_k, Lambda_k = [], []
    for k in range(K):
        H_ = np.expand_dims(H[k, :], axis=0)
        W_k.append(tensorW[:, :, k].T)
        tensorW_ = np.expand_dims(tensorW[:, :, k], axis=2)
        Lambda_k.append(convModel(tensorW_, H_))

    results = {}
    results['tensorW'] = tensorW
    results['H'] = H
    results['S_hat'] = Lambda_V
    results['S_hat_k'] = Lambda_k
    results['W_k'] = W_k
    results['costFunc'] = costFunc
    if record_updates:
        results['updates_H'] = updates_H

    return results