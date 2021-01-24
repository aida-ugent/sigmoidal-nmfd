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

from .experiment_nmfdsigmoid_on_enst import *

import argparse
import csv
import matplotlib
matplotlib.use('Agg')  # Uncomment when running on a headless server without a display to plot on
import matplotlib.pyplot as plt
import numpy as np
import os
import textwrap

import nmfd.display
from scripts.run_nmfd_sparsity import run_nmfd_sparsity

from multiprocessing import Process


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Ablation study script for sparse NMFD: run sparse NMDF with a warm-up stage.',
        epilog=textwrap.dedent('''\
            Runs the NMFD algorithm with L1 regularization and warm-up on the ENST dataset.
        '''))
    parser.add_argument('--dir-enst', required=True,
                        help='path to the ENST dataset. ')
    parser.add_argument('--dir-out', required=True,
                        help='output directory.\n'
                             'The script will create a subdirectory for each ablation study.')
    parser.add_argument('--tracklist',
                        default='resources/tracklists/tracklist_enst_allphrases.csv',
                        help='csv file containing a list of paths (relative to dir-enst) of files to be analyzed. '
                             'Defaults to all ENST phrases.')
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='Parallel processing of each ablation study using Python subprocesses.')
    args = parser.parse_args()

    files = []
    with open(args.tracklist) as tracklist_file:
        reader = csv.reader(tracklist_file)
        for l in reader:
            if l[0][0] == '#':
                continue
            files.append(tuple(os.path.join(args.dir_enst, li) for li in l))

    # ------------------------------------
    # Specify the experiment parameters
    # ------------------------------------
    sparsities = [
        0.01,
        0.1,
        1.0,
    ]

    warmups = [
        # False,  # Uncomment to evaluate 'default' setting (is also evaluated in experiment_nmfdsigmoid_... script)
        True
    ]

    def sparsity_to_path(sp):
        return f'{sp}'.replace('.', '')

    # Create output subdirectories
    experiment_params = []
    for sparsity in sparsities:
        for do_warmup in warmups:
            dir_out_ = os.path.join(args.dir_out, 'warmup' if do_warmup else 'default', sparsity_to_path(sparsity))
            if not os.path.exists(dir_out_):
                os.makedirs(dir_out_)
            experiment_params.append((do_warmup, sparsity, dir_out_))

    # ------------------------------------
    # Perform each experiment
    # ------------------------------------
    processes = []
    for do_warmup, sparsity, dir_out_ in experiment_params:

        def experiment_subprocess():
            for idx, (path_to_wav, path_to_annot) in enumerate(files):

                path_to_npz = f'{nmfd.display.filename_to_compact_string(path_to_wav)}.npz'
                print(f'({idx + 1}/{len(files)}) {path_to_npz}')

                # Crop the audio file
                path_to_wav_cropped = os.path.join(*get_cropped_filename(path_to_wav))
                if not os.path.exists(path_to_wav_cropped):
                    print('\tcropping the wav file (exclude the last hit)...')
                    if args.is_parallel:
                        raise Exception('Not allowed to crop file in parallel mode, as this is thread unsafe!')
                    crop_enst_file(path_to_wav, path_to_annot)

                # Get the number of components
                num_components = get_groundtruth_num_components_from_annot(path_to_annot)

                if path_to_npz in os.listdir(dir_out_):
                    print(f'\talready performed experiment {"warmup" if do_warmup else "no_warmup"} {sparsity} on this file...')
                else:
                    print(f'\tperforming ablation {"warmup" if do_warmup else "no_warmup"} {sparsity}...')
                    run_nmfd_sparsity(
                        path_to_wav_cropped,
                        num_components,
                        sparsity=sparsity,
                        out_dir=dir_out_,
                        is_plot=False,
                        annotation_file=path_to_annot,
                        do_warmup=do_warmup,
                    )
        if args.parallel:
            p = Process(target=experiment_subprocess)
            p.start()
            processes.append(p)
        else:
            experiment_subprocess()

    for p in processes:
        p.join()

    # ---------------------
    # Show results
    # ---------------------

    def plot_threshold_curve(results, label=''):
        r = np.array(
            [r['_coverage_evolution_theta'] for r in results])  # dimensions len(results), len(thresholds), 2
        r = np.average(r, axis=0)
        plt.plot(np.linspace(0.1, 0.9, 9), r, '-o', label=label)

    results = {}
    for do_warmup, sparsity, dir_out_ in experiment_params:
        experiment_title = f'{"warmup" if do_warmup else "default"} {sparsity}'
        results_ = gather_results(dir_out_, substitute_enst_dir=args.dir_enst)
        results[experiment_title] = results_

        print('--------------------------')
        visualize_results(results_, results_, None, xlabel=experiment_title, ylabel=experiment_title)

