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

from .NMFtoolbox.convModel import convModel_optimized as convModel_
from .NMFtoolbox.shiftOperator import shiftOperator


def sigmoid(x, scale=1.0, offset=0.0):
    x_ = np.clip(x, -50./scale, 50./scale) # Clip logit values to avoid overflow or underflow
    return 1 / (1 + np.exp(-scale*(x_ - offset)))


def sigmoid_inv(x):
    return np.log(x / (1-x))


def convModel(tensorW, H, a, sigmoid_scale=1.0, sigmoid_offset=0.0):
    a_ = sigmoid(a, 1)[:, np.newaxis]
    return convModel_(tensorW, a_*sigmoid(H, sigmoid_scale, sigmoid_offset))


def update_term_H_KL(X, H, tensorW, A):
    L_tau, N, K = tensorW.shape
    K, T = H.shape

    Lambda = convModel(tensorW, H, A) + 1e-18
    X_Lambda_ = X * Lambda ** -1
    dLdX = (1 - X_Lambda_)
    tensorW_T = tensorW.transpose((2, 1, 0))
    A_sigmoid = sigmoid(A, 1)
    H_sigmoid = sigmoid(H)
    one_minus_H_sigmoid = sigmoid(-H)  # More numerically stable for sigmoid(H) ~ 1

    update = np.zeros_like(H)

    for tau in range(L_tau):
        dLdX_shifted = dLdX[:, tau:T + tau]
        update += np.sum(
            dLdX_shifted[np.newaxis, :, :] *
            tensorW_T[:, :, tau, np.newaxis] *
            A_sigmoid[:, np.newaxis, np.newaxis] *
            (H_sigmoid * one_minus_H_sigmoid)[:, np.newaxis, :],
            axis=(1,)
        )

    return [update]


def update_term_W_KL(X, H, tensorW, A):
    L_tau, N, K = tensorW.shape

    update = np.ones_like(tensorW)
    Lambda = convModel(tensorW, H, A) + 1e-18

    X_Lambda_ = (X + 1e-18) * Lambda ** -1
    X_Lambda_ = X_Lambda_[:, :-L_tau + 1]
    ones_ = np.ones_like(X_Lambda_)
    A_sigmoid = sigmoid(A, 1)
    H_sigmoid = A_sigmoid[:, np.newaxis] * sigmoid(H)

    for tau in range(L_tau):
        transpH = shiftOperator(H_sigmoid, tau).T
        numerator = (X_Lambda_) @ transpH
        denominator = ones_ @ transpH
        update[tau, :, :] = numerator / denominator

    return update


def update_term_amplitude_KL(X, H, tensorW, A):
    L_tau, N, K = tensorW.shape
    Lambda = convModel(tensorW, H, A) + 1e-18
    X_Lambda_ = X * Lambda ** -1
    dLdX = (1 - X_Lambda_)
    # H_sigmoid = sigmoid(H)
    A_sigmoid = sigmoid(A, 1)
    one_minus_A_sigmoid = sigmoid(-A, 1)
    update = np.zeros((K))

    for k in range(K):
        W_k = tensorW[:, :, k, None]
        H_k = H[k, None, :]
        update[k] = (
                A_sigmoid[k] * one_minus_A_sigmoid[k] *
                np.sum(convModel(W_k, H_k, [0]) * dLdX)
        )
    update /= np.abs(update).max()
    return update


def get_mu_k(H_logit, alpha=0.5):
    H = sigmoid(H_logit, 1, 0)
    mu = sigmoid_inv((alpha * np.max(H, axis=1) + (1 - alpha) * np.min(H, axis=1)))
    return mu[:, np.newaxis]


def loss_function(X, H, tensorW, A, penalty_weight=1.0):

    # Spectrogram loss
    Lambda = convModel(tensorW, H, A) + 1e-18
    Lambda = Lambda
    loss_spectrogram = np.sum(X * (np.log(X + 1e-18) - np.log(Lambda)) + Lambda - X)

    # Penalty on G
    mu_k = get_mu_k(H, alpha=0.5)
    loss_activations = penalty_weight * np.sum(np.exp(-((H - mu_k) / 2) ** 2), axis=1)

    return loss_spectrogram + np.sum(loss_activations)


def update_penalty_wedge_H(H_logit, wedge_wiggle=False):
    if wedge_wiggle:
        alpha = np.random.uniform(low=0.05, high=0.25, size=H_logit.shape[0])
    else:
        alpha = 0.1
    mu = get_mu_k(H_logit, alpha)
    x_ = H_logit - mu
    dPdH = - np.exp(- (x_ / 2) ** 2) * x_

    return dPdH


def nmfd_sigmoid(S, parameter=None, paramConstr=None):

    # Initialize shape variables
    N, T_length = S.shape
    L_tau = parameter['numTemplateFrames']
    K = parameter['numComp']

    # Initialize tensorW
    tensorW = np.zeros((L_tau, N, K))
    for k in range(K):
        tensorW[:, :, k] = parameter['initW'][k].T

    # Initialize H
    H = deepcopy(parameter['initH'])
    A = deepcopy(parameter['initA'])

    # Initialize target spectrogram
    V_tmp = S / np.max(S)

    # Record the updates per iteration -- for logging and debugging purposes
    record_updates = parameter.get('record_updates', False)
    updates_H = []

    # List to log cost function values at each iteration
    costFunc = []

    # Perform ablation study if specified
    ablation = parameter.get('ablation', None)
    if not ablation is None:
        print(f'Performing ablation study {ablation}.')
    else:
        # Ablation is None, we use the following setting, as evaluated on ENST dataset:
        # low loss per timestep, while maintaining a low mean and max similarity between activations
        ablation = 'no_anneal_with_wiggle'

    gamma = 1.0 if '_01' not in ablation else 0.1

    # -----------------------------------------
    # OPTIMIZATION STRATEGIES AND PARAMETERS
    # -----------------------------------------
    NUM_ITER = 240

    # Warm-up stage parameters
    # ---
    H_penalty_weight_init_cycle = [0.0] * 30
    H_learning_rate_init_cycle = [5e-1] * 30

    if 'no_warmup' in ablation:
        H_penalty_weight_init_cycle = [gamma] * 30
        H_learning_rate_init_cycle = [2e-1] * 30

    # Finalization phase parameters
    # ---
    H_penalty_weight_fin_cycle = [1e0] * 30
    H_learning_rate_fin_cycle = [1e-1] * 30

    n_anneal_iter = 3
    n_before_finalize = 30 + n_anneal_iter * 60

    # Explore-and-converge stage parameters
    # ---
    H_learning_rate_cycle = ([2e-1] * 60)

    # Enable/disable gamma periodically during explore-and-converge
    if 'no_anneal_' in ablation:
        H_penalty_weight_cycle = ([gamma] * 60)
    else:
        H_penalty_cycle_grow = [gamma] * 30
        H_penalty_cycle_stabilise = [0.0] * 30
        H_penalty_weight_cycle = H_penalty_cycle_grow + H_penalty_cycle_stabilise

    # Optimization technique: Move around mu_k or not
    if 'with_wiggle' in ablation:
        wedge_wiggle_cycle = [True] * n_before_finalize + [False] * 30
    else:
        wedge_wiggle_cycle = [False] * n_before_finalize + [False] * 30

    # Put everything together
    # ---
    H_penalty_weight_cycle = H_penalty_weight_init_cycle + H_penalty_weight_cycle * n_anneal_iter + H_penalty_weight_fin_cycle
    H_learning_rate_cycle = H_learning_rate_init_cycle + H_learning_rate_cycle * n_anneal_iter + H_learning_rate_fin_cycle
    A_learning_rate_cycle = [2e-2] * NUM_ITER

    if 'fixed_lr' in ablation:
        # Constant learning rate for G
        H_learning_rate_cycle = [1e-1 if 'fixed_lr_small' in ablation else 2e-1] * NUM_ITER

    H_penalty_weight_cycle = np.array(H_penalty_weight_cycle)
    H_learning_rate_cycle = np.array(H_learning_rate_cycle)
    A_learning_rate_cycle = np.array(A_learning_rate_cycle)

    assert(NUM_ITER
           == len(H_penalty_weight_cycle)
           == len(H_learning_rate_cycle)
           == len(A_learning_rate_cycle)
           == len(wedge_wiggle_cycle))

    # -----------------------------------------
    # OPTIMIZATION LOOP
    # -----------------------------------------

    def _normalize_tensorW(tensorW, H):
        tensorW /= np.max(tensorW, axis=(0, 1), keepdims=True)

    _normalize_tensorW(tensorW, H)

    for iter_ in range(NUM_ITER):

        H_learning_rate = H_learning_rate_cycle[iter_ % len(H_learning_rate_cycle)]     # eta_G
        A_learning_rate = A_learning_rate_cycle[iter_ % len(A_learning_rate_cycle)]     # eta_a
        H_penalty_weight = H_penalty_weight_cycle[iter_ % len(H_penalty_weight_cycle)]  # gamma

        wedge_wiggle = wedge_wiggle_cycle[iter_ % len(wedge_wiggle_cycle)]  # Move mu_k around or not?

        _normalize_tensorW(tensorW, H)

        # --------- UPDATE H (onset functions) --------------
        # Update of H wrt first objective (spectrogram)
        update_H = update_term_H_KL(V_tmp, H, tensorW, A)
        update_H = np.sum(update_H, axis=0)

        # Update of H wrt second objective (saturation)
        update_H += H_penalty_weight * update_penalty_wedge_H(H, wedge_wiggle=wedge_wiggle)
        if not 'no_gradient_normalization' in ablation:
            update_H /= np.max(np.abs(update_H), axis=1)[:, np.newaxis]

        # Apply the gradient descent update
        H = H - H_learning_rate * update_H

        # --------- UPDATE W (spectrogram templates) --------
        update_W = update_term_W_KL(V_tmp, H, tensorW, A)
        tensorW *= update_W
        _normalize_tensorW(tensorW, H)

        # --------- UPDATE A (sigmoidal amplitude) ----------
        update_A = update_term_amplitude_KL(V_tmp, H, tensorW, A)
        if not 'no_gradient_normalization' in ablation:
            update_A /= np.max(np.abs(update_A))

        A = A - A_learning_rate * update_A

        # store the divergence with respect to the target spectrogram
        costFunc.append(loss_function(V_tmp, H, tensorW, A))

        if record_updates:
            S_hat = convModel(tensorW, H, A)
            updates_record = {
                'iter_': iter_,
                'S': np.copy(V_tmp),
                'S_hat': S_hat,
                'sigmoid_scale': 1.0,
                'sigmoid_offset': 0,
                'H': np.copy(sigmoid(H)),
                'H_logit': np.copy(H),
                'W': np.copy(tensorW),
                'A': np.copy(sigmoid(A, 1)),
                'A_logit': np.copy(A),
                'W_update': update_W,
                'H_update': update_H,
            }
            updates_H.append(updates_record)

    # Overall approximation
    Lambda_V = convModel(tensorW, H, A)

    # Per-component approximations
    W_k, Lambda_k = [], []
    for k in range(K):
        H_ = np.expand_dims(H[k, :], axis=0)
        W_k.append(tensorW[:, :, k].T)
        tensorW_ = np.expand_dims(tensorW[:, :, k], axis=2)
        Lambda_k.append(convModel(tensorW_, H_, [A[k]]))

    results = {}
    results['tensorW'] = tensorW
    results['H'] = sigmoid(H)
    results['H_logit'] = H
    results['sigmoid_scale'] = 1.0
    results['sigmoid_offset'] = 0.0
    results['A'] = sigmoid(A, 1)
    results['A_logit'] = A
    results['S_hat'] = Lambda_V
    results['S_hat_k'] = Lambda_k
    results['W_k'] = W_k
    results['costFunc'] = costFunc
    if record_updates:
        results['updates_H'] = updates_H

    return results
