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

import argparse
import csv
import librosa
import matplotlib
# matplotlib.use('Agg')  # Uncomment when running on a headless server without a display to plot on
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf
import textwrap

from sklearn.metrics.pairwise import cosine_similarity

import nmfd.display
from scripts.run_nmfd_vanilla import run_nmfd_vanilla
from scripts.run_nmfd_sigmoid import run_nmfd_sigmoid
from scripts.run_nmfd_sparsity import run_nmfd_sparsity

# ======================================================================================================================
# === Utilities to crop the ENST audio files. ==========================================================================
# ======================================================================================================================

def get_last_hit_from_onset_file(path_to_annot):
    ''' Get the onset time of the last onset for an ENST annotation file.

    :param path_to_annot: Path to the annotation file.
    :return: The onset time of the last drum hit, in seconds, for this annotation file.
    '''

    with open(path_to_annot) as f:
        reader = csv.reader(f, delimiter=' ')
        hits = [float(l[0]) for l in reader]
    return np.max(hits)


def get_cropped_filename(path_to_wav):
    dir_, basename = os.path.split(path_to_wav)
    return os.path.join(dir_, 'cropped/'), basename


def crop_enst_file(path_to_wav, path_to_annot):

    T = get_last_hit_from_onset_file(path_to_annot)

    y, sr = librosa.load(path_to_wav, sr=44100, mono=False, dtype='float64')
    y[:, int(sr * (T - 0.1)): int(sr * T)] *= np.linspace(1, 0, int(sr * T) - int(sr * (T - 0.1)))
    y[:, int(sr * T):] *= 0
    y = y[:, : int(sr * (T + 0.2))]

    out_dir, basename = get_cropped_filename(path_to_wav)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    output_file = os.path.join(out_dir, f'{os.path.splitext(basename)[0]}.wav')
    sf.write(output_file, y.T, sr)


# ======================================================================================================================
# === Utilities to load info from the ENST annotation files ============================================================
# ======================================================================================================================

def get_groundtruth_num_components_from_annot(path_to_annot):

    with open(path_to_annot) as f:
        reader = csv.reader(f, delimiter=' ')
        instrs = {}
        for l in reader:
            if l[1] not in instrs:
                instrs[l[1]] = 1
            else:
                instrs[l[1]] += 1
        return len(instrs.keys())


# ======================================================================================================================
# === Evaluation =======================================================================================================
# ======================================================================================================================


# ==== MAE ====================================
def metric_spectrogram_mae(S, S_hat):
    return np.mean(np.abs(S - S_hat))


# ==== Activations similarity =================
def running_mean(x, N=11):
    return np.convolve(x, np.ones((N,))/N, mode='same') #[(N-1):]


def metric_activations_similarity(H):
    if H.shape[0] == 1:
        # raise Exception('Cannot measure similarity of one component!')
        return [0,0]
    H_ = np.copy(H)
    for j in range(H_.shape[0]):
        H_[j] = running_mean(H_[j]) + 1e-52
    sim = cosine_similarity(H_)
    upper_triangle_sims = []
    for i in range(0,H.shape[0]):
        for j in range(i+1,H.shape[0]):
            # if H_[i].max() <= 1e-52 and H_[j].max() <= 1e-52:
            #     upper_triangle_sims.append(1.0)
            # else:
            upper_triangle_sims.append(sim[i,j])
    return sorted(upper_triangle_sims)

# ==== Peakedness =============================
def compand(h, a):
    h_max = h.max()
    return ((h / (h_max + 1e-18)) ** a ) * h_max


def hwr(h):
    return (h - running_mean(h, N=11)).clip(0)


def metric_peakedness_hwr_ratio(h, a=3.0):
    h_hwr = compand(hwr(compand(h, a)), 1/a)
    return np.sum(h_hwr) / np.sum(h)


# ==== Number of zero rows ====================
# (not used in final paper)

def metric_count_number_of_zero_rows(H):
    h_sum = np.sum(H, axis=1)
    return np.sum(h_sum < 1e-52) / len(h_sum)


# ==== Overall onset coverage =================
def get_hits_from_annotation_file(annotation_file):
    with open(annotation_file) as csvfile:
        reader = csv.reader(csvfile)
        hits = list(tuple(l[0].split(' ')) for l in reader)
        hits, instrs = zip(*[(float(h[0]), h[1]) for h in hits])
    return list(hits), list(instrs)


def get_peaks_for_W(W_k):
    w = np.sum(W_k, axis=0)
    w_mean = np.mean(w)
    peak = np.where(w > w_mean)[0][0]
    return [peak]


def get_peaks_from_H(H, W, aggregate=False, consider_W_peaks=True, delta=0.1):
    peaks = []

    pre_max, post_max = 5, 5
    pre_avg, post_avg = 10, 10
    delta, wait = delta, 10

    for k in range(H.shape[0]):
        h = H[k]
        if consider_W_peaks:
            W_k = W[k]
            taus = get_peaks_for_W(W_k)
        else:
            taus = [0]

        # Perform peak picking relative to the maximum value of this ODF
        delta_k = delta * max(np.max(h), 0.2)
        p = list(librosa.util.peak_pick(h, pre_max, post_max, pre_avg, post_avg, delta_k, wait))
        if not aggregate:
            for tau in taus:
                peaks.append([pi + tau for pi in p])
        else:
            for tau in taus:
                peaks.extend([pi + tau for pi in p])

    if aggregate:
        peaks = sorted(peaks)

    return peaks


def pseudo_precision_recall(hits, peaks):
    last_peak = peaks[-1] if len(peaks) > 0 else 0
    U = np.zeros((2, max(hits[-1] + 1, last_peak + 1)))
    U[0, hits] = 1
    U[1, peaks] = 1

    tol_before = 5  # 30 ms tolerance ~ 5 frames
    tol_after = 5

    h_gt = U[0]
    h_nmf = U[1]
    h_gt_detected = np.zeros_like(h_gt)
    h_nmf_detected = np.zeros_like(h_nmf)

    for t in range(U.shape[1]):
        t_min, t_max = max(t - tol_before, 0), min(t + tol_after, U.shape[1])
        if h_gt[t] == 1:
            if np.any(h_nmf[t_min:t_max]):
                h_gt_detected[t] = 1
        if h_nmf[t] == 1:
            if np.any(h_gt[t_min:t_max]):
                h_nmf_detected[t] = 1

    # total ground truth (N), not detected ground truth (false negatives), p, tp (again)
    N = np.sum(h_gt)
    fn = N - np.sum(h_gt_detected)
    p = np.sum(h_nmf)
    tp = np.sum(h_nmf_detected)
    return N, fn, p, tp


def metric_overall_onset_precision_recall(path_to_npz, path_to_annot, theta=0.1):
    H = np.load(path_to_npz)['H']
    W = np.load(path_to_npz)['W']
    hits, instrs = get_hits_from_annotation_file(path_to_annot)
    # The spectrogram was padded with 1024 samples upon loading.
    # Offset the annotations by this much as well.
    # sr 44100, hop size 256
    hits = [int((1024 + t * 44100) / 256) for t in hits]
    peaks = get_peaks_from_H(H, W, aggregate=True, consider_W_peaks=True, delta=theta)

    n_hits, fn, p, tp = pseudo_precision_recall(hits, peaks)
    precision = tp / p if p > 0 else 0 # how many selected items are relevant
    recall = (n_hits - fn) / n_hits # how many relevant items are selected

    return precision, recall


def fmeasure(precision, recall):
    return (2*precision*recall) / (precision + recall + (1 if precision == recall == 0 else 0))


# ==== Process results for one file ===========
def evaluate_result(filepath, substitute_enst_dir=None):
    retval = {}
    result = np.load(filepath)
    H = result['H']

    K, N, L_tau = result['W'].shape
    retval['MAE'] = metric_spectrogram_mae(result['S_stft'], result['S_out'][:, :-L_tau + 1])

    annotation_file_path = result['annotation_file'][0]
    if not substitute_enst_dir is None:
        path_relative_idx = annotation_file_path.find('drummer_')
        annotation_file_path = os.path.join(substitute_enst_dir, annotation_file_path[path_relative_idx:])

    precision, recall = metric_overall_onset_precision_recall(filepath, annotation_file_path)
    precision5, recall5 = metric_overall_onset_precision_recall(filepath, annotation_file_path, theta=0.5)
    retval['Onset coverage precision'] = precision
    retval['Onset coverage recall'] = recall
    retval['Onset coverage F-measure'] = fmeasure(precision, recall)
    retval['Onset coverage precision, high threshold'] = precision5
    retval['Onset coverage recall, high threshold'] = recall5
    retval['Onset coverage F-measure, high threshold'] = fmeasure(precision5, recall5)

    retval['Activations similarity (min)'] = metric_activations_similarity(H)[0]
    retval['Activations similarity (mean)'] = np.mean(metric_activations_similarity(H))
    retval['Activations similarity (max)'] = metric_activations_similarity(H)[-1]

    peakednesses = list(metric_peakedness_hwr_ratio(h) for h in list(H))
    retval['Peakedness (min)'] = np.min(peakednesses)
    retval['Peakedness (mean)'] = np.mean(peakednesses)
    retval['Peakedness (mean, no zero rows)'] = np.mean([p for p,h in zip(peakednesses, list(H)) if np.sum(h) > 1e-52])
    retval['Peakedness (max)'] = np.max(peakednesses)

    retval['Zero rows ratio (mean)'] = metric_count_number_of_zero_rows(H)

    try:
        retval['Loss'] = result['loss'][-1] / H.shape[1]
    except IndexError:
        retval['Loss'] = result['loss'] / H.shape[1]

    retval['_filepath'] = filepath
    retval['_sourcefile'] = result['source_file']
    retval['_H'] = H

    retval['_coverage_evolution_theta'] = [
        fmeasure(*metric_overall_onset_precision_recall(filepath, annotation_file_path, theta=d))
        for d in np.linspace(0.1,0.9,9)
    ]

    return retval


# ==== Process results for all files ==========
def gather_results(out_dir, substitute_enst_dir=None):
    results = []
    for f in sorted(os.listdir(out_dir)):
        if f.endswith('.npz'):
            results.append(evaluate_result(os.path.join(out_dir, f), substitute_enst_dir=substitute_enst_dir))

    print(f'Results gathered for {len(results)} files.')
    return results


def visualize_results(results_1, results_2, colors, xlabel='', ylabel='', is_plot=False):
    results_zipped = {}
    for r1, r2 in zip(list(results_1), list(results_2)):
        for k in r1.keys():
            if k not in results_zipped:
                results_zipped[k] = []
            results_zipped[k].append((r1[k], r2[k]))

    plt.figure(figsize=(10, 10))
    for i, (k, v) in enumerate(results_zipped.items()):
        if k.startswith('_'):
            continue
        x, y = list(zip(*v))
        print('')
        print(k)
        print('---')
        print(f'{xlabel}: {np.mean(x):.3f} ({np.std(x):.3f})')
        print(f'{ylabel}: {np.mean(y):.3f} ({np.std(y):.3f})')
        print('===============================================')
        if is_plot:
            plt.scatter(x, y, c=colors)
            min_ = np.min((np.min(x), np.min(y)))
            max_ = np.max((np.max(x), np.max(y)))
            min__ = min_ - (max_ - min_) * 0.1
            max__ = max_ + (max_ - min_) * 0.1
            plt.xlim(min__, max__)
            plt.ylim(min__, max__)
            plt.plot(np.linspace(min__, max__, 100), np.linspace(min__, max__, 100), alpha=0.5)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(k)
            plt.show()


# ======================================================================================================================
# === Main script ======================================================================================================
# ======================================================================================================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Evaluation script for sigmoidal NMFD.',
        epilog=textwrap.dedent('''\
            Runs the algorithm on the ENST dataset, saves the decompositions to disk,
            and evaluates the results using the metrics defined in the paper.
            
            Download the ENST dataset at:
            http://www.tsi.telecom-paristech.fr/aao/en/2010/02/19/enst-drums-an-extensive-audio-visual-database-for-drum-signals-processing/.
        '''))
    parser.add_argument('--dir-enst', required=True,
                        help='path to the ENST dataset. ')
    parser.add_argument('--dir-out', required=True,
                        help='output directory.\n'
                             'The script will create a subdirectory for each baseline and the evaluated sigmoidal model. '
                             'In these, the decompositions for each file in the ENST database are saved.')
    parser.add_argument('--tracklist',
                        default='resources/tracklists/tracklist_enst_allphrases.csv',
                        help='csv file containing a list of paths (relative to dir-enst) of files to be analyzed. '
                             'Defaults to all ENST phrases.')
    args = parser.parse_args()

    files = []
    with open(args.tracklist) as tracklist_file:
        reader = csv.reader(tracklist_file)
        for l in reader:
            if l[0][0] == '#':
                continue
            files.append(tuple(os.path.join(args.dir_enst, li) for li in l))

    # Create output directories
    dir_out_vanilla = os.path.join(args.dir_out, 'vanilla')
    dir_out_sigmoid = os.path.join(args.dir_out, 'sigmoid')
    dir_out_sigmoid_simpl = os.path.join(args.dir_out, 'sigmoid_simp')
    dir_out_sparsity = os.path.join(args.dir_out, 'sparsity_1')
    dir_out_sparsity01 = os.path.join(args.dir_out, 'sparsity_01')
    dir_out_sparsity001 = os.path.join(args.dir_out, 'sparsity_001')
    if not os.path.exists(dir_out_vanilla):
        os.mkdir(dir_out_vanilla)
    if not os.path.exists(dir_out_sigmoid):
        os.mkdir(dir_out_sigmoid)
    if not os.path.exists(dir_out_sigmoid_simpl):
        os.mkdir(dir_out_sigmoid_simpl)
    if not os.path.exists(dir_out_sparsity):
        os.mkdir(dir_out_sparsity)
    if not os.path.exists(dir_out_sparsity01):
        os.mkdir(dir_out_sparsity01)
    if not os.path.exists(dir_out_sparsity001):
        os.mkdir(dir_out_sparsity001)

    for idx, (path_to_wav, path_to_annot) in enumerate(files):

        path_to_npz = f'{nmfd.display.filename_to_compact_string(path_to_wav)}.npz'
        print(f'({idx + 1}/{len(files)}) {path_to_npz}')

        # Crop the audio file
        path_to_wav_cropped = os.path.join(*get_cropped_filename(path_to_wav))
        if not os.path.exists(path_to_wav_cropped):
            print('\tcropping the wav file (exclude the last hit)...')
            crop_enst_file(path_to_wav, path_to_annot)

        # Get the number of components
        num_components = get_groundtruth_num_components_from_annot(path_to_annot)

        # Perform NMFD, vanilla
        if path_to_npz in os.listdir(dir_out_vanilla):
            print('\talready performed vanilla NMFD...')
        else:
            print('\tperforming vanilla NMFD...')
            run_nmfd_vanilla(
                path_to_wav_cropped,
                num_components,
                out_dir=dir_out_vanilla,
                is_plot=False,
                annotation_file=path_to_annot,
            )

        # Perform NMFD, sigmoid
        if path_to_npz in os.listdir(dir_out_sigmoid):
            print('\talready performed sigmoid NMFD...')
        else:
            print('\tperforming sigmoid NMFD...')
            run_nmfd_sigmoid(
                path_to_wav_cropped,
                num_components,
                out_dir=dir_out_sigmoid,
                is_plot=False,
                annotation_file=path_to_annot,
            )

        # Perform NMFD, sigmoid (simplified)
        if path_to_npz in os.listdir(dir_out_sigmoid_simpl):
            print('\talready performed simplified sigmoid NMFD...')
        else:
            print('\tperforming simplified sigmoid NMFD...')
            run_nmfd_sigmoid(
                path_to_wav_cropped,
                num_components,
                out_dir=dir_out_sigmoid_simpl,
                is_plot=False,
                annotation_file=path_to_annot,
                ablation='fixed_lr_no_anneal_without_wiggle',
            )

        # Perform NMFD, L1 sparsity
        if path_to_npz in os.listdir(dir_out_sparsity):
            print('\talready performed L1 NMFD...')
        else:
            print('\tperforming L1 NMFD...')
            run_nmfd_sparsity(
                path_to_wav_cropped,
                num_components,
                out_dir=dir_out_sparsity,
                is_plot=False,
                annotation_file=path_to_annot,
                sparsity=1.0
            )

        # Perform NMFD, L1 sparsity
        if path_to_npz in os.listdir(dir_out_sparsity01):
            print('\talready performed L1 NMFD...')
        else:
            print('\tperforming L1 NMFD...')
            run_nmfd_sparsity(
                path_to_wav_cropped,
                num_components,
                out_dir=dir_out_sparsity01,
                is_plot=False,
                annotation_file=path_to_annot,
                sparsity=0.1
            )

        # Perform NMFD, L1 sparsity
        if path_to_npz in os.listdir(dir_out_sparsity001):
            print('\talready performed L1 NMFD...')
        else:
            print('\tperforming L1 NMFD...')
            run_nmfd_sparsity(
                path_to_wav_cropped,
                num_components,
                out_dir=dir_out_sparsity001,
                is_plot=False,
                annotation_file=path_to_annot,
                sparsity=0.01
            )

    results_sigmoid = gather_results(dir_out_sigmoid)
    results_sigmoid_simpl = gather_results(dir_out_sigmoid_simpl)
    results_vanilla = gather_results(dir_out_vanilla)
    results_sparsity = gather_results(dir_out_sparsity)
    results_sparsity_01 = gather_results(dir_out_sparsity01)
    results_sparsity_001 = gather_results(dir_out_sparsity001)

    def get_color_difficulty(filename):
        if 'simple' in filename:
            return 'g'
        elif 'complex' in filename:
            return 'r'
        else:
            raise Exception('Mystery difficulty!')

    colors = [get_color_difficulty(r['_sourcefile'][0]) for r in results_sigmoid]

    print('--------------------------')
    visualize_results(results_vanilla, results_sigmoid, colors, xlabel='Default NMFD', ylabel='Sigmoid NMFD')
    print('--------------------------')
    visualize_results(results_sparsity, results_sparsity_01, colors, xlabel='Sparse NMFD lambda=1.0', ylabel='lambda=0.1')
    print('--------------------------')
    visualize_results(results_vanilla, results_sparsity_001, colors, xlabel='Default NMFD', ylabel='lambda=0.01')
    print('--------------------------')

    def plot_threshold_curve(results, label='', **kwargs):
        r = np.array([r['_coverage_evolution_theta'] for r in results]) # dimensions len(results), len(thresholds), 2
        r = np.average(r, axis=0)
        plt.plot(np.linspace(0.1,0.9,9), r, '-o', label=label, **kwargs)

    plt.figure()
    plot_threshold_curve(results_vanilla, label='NMFD')
    plot_threshold_curve(results_sparsity_001, label='NMFD, lambda=0.01')
    plot_threshold_curve(results_sparsity_01, label='NMFD, lambda=0.1')
    plot_threshold_curve(results_sparsity, label='NMFD, lambda=1.0')
    plot_threshold_curve(results_sigmoid, label=r'Sigmoid NMFD (strat. 2, $\gamma$=0.1)', c='C4', ls='--')
    plot_threshold_curve(results_sigmoid_simpl, label=r'Sigmoid NMFD (strat. 0, $\gamma$=1.0, $\eta_G$=0.2)', c='C4')
    plt.legend()
    plt.xlabel(r'Peak picking threshold $\theta$')
    plt.ylabel('F-measure')
    plt.tight_layout()
    plt.show()

