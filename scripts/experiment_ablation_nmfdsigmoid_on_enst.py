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
# Uncomment the line below when running on a headless server without a display to plot on;
# comment when you want to see plots
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import textwrap

import nmfd.display
from scripts.run_nmfd_sigmoid import run_nmfd_sigmoid

from multiprocessing import Process


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Evaluation script for sigmoidal NMFD: ablation study.',
        epilog=textwrap.dedent('''\
            Runs the sigmoid NMFD algorithm on the ENST dataset in different ablation configurations.
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

    ablations = [
        # Test out the different strategies for optimizing the model
        'no_anneal_without_wiggle',       # Strategy 0, gamma = 1.0
        'no_anneal_without_wiggle_01',    # Strategy 0, gamma = 0.1
        'no_anneal_with_wiggle',          # Strategy 2, gamma = 1.0
        'no_anneal_with_wiggle_01',       # Strategy 2, gamma = 0.1
        'anneal_without_wiggle',          # Strategy 1, gamma = 1.0
        'anneal_without_wiggle_01',       # Strategy 1, gamma = 0.1
        'anneal_with_wiggle',             # Strategy 3, gamma = 1.0
        'anneal_with_wiggle_01',          # Strategy 3, gamma = 0.1
        # Ablation: no gradient normalization
        'no_anneal_without_wiggle_no_gradient_normalization',       # No gradient norm strat 0 gamma = 1.0
        'no_anneal_with_wiggle_01_no_gradient_normalization',       # No gradient norm strat 2 gamma = 0.1
        # Ablation: no warmup
        'no_warmup_no_anneal_without_wiggle', # Disable warmup, strategy 0 gamma = 1.0
        'no_warmup_anneal_without_wiggle',    # Disable warmup, strategy 1 gamma = 1.0 (not reported in paper)
        'no_warmup_no_anneal_with_wiggle',    # Disable warmup, strategy 2 gamma = 1.0 (not reported in paper)
        'no_warmup_anneal_with_wiggle',       # Disable warmup, strategy 3 gamma = 1.0 (not reported in paper)
        'no_warmup_no_anneal_without_wiggle_01',    # Disable warmup, strategy 0 gamma = 0.1 (not reported in paper)
        'no_warmup_anneal_without_wiggle_01',       # Disable warmup, strategy 1 gamma = 0.1 (not reported in paper)
        'no_warmup_no_anneal_with_wiggle_01',       # Disable warmup, strategy 2 gamma = 0.1
        'no_warmup_anneal_with_wiggle_01',          # Disable warmup, strategy 3 gamma = 0.1 (not reported in paper)
        # Ablation: fixed learning rate for G
        'fixed_lr_no_anneal_without_wiggle',        # Strat 0 gamma = 1.0 without LR changes in G
        'fixed_lr_no_anneal_with_wiggle_01',        # Best performing model without LR changes in G
        'fixed_lr_small_no_anneal_without_wiggle',  # Strat 0 gamma = 1.0 without LR changes in G, and w/ smaller LR
        'fixed_lr_small_no_anneal_with_wiggle_01',  # Best performing model without LR changes in G, and w/ smaller LR
    ]

    # Create output subdirectories
    for abl in ablations:
        dir_out_ = os.path.join(args.dir_out, abl)
        if not os.path.exists(dir_out_):
            os.mkdir(dir_out_)

    # Perform each ablation
    processes = []
    for abl in ablations:
        dir_out_abl = os.path.join(args.dir_out, abl)
        def ablation_study_subprocess():
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

                if path_to_npz in os.listdir(dir_out_abl):
                    print(f'\talready performed ablation {abl} on this file...')
                else:
                    print(f'\tperforming ablation {abl}...')
                    run_nmfd_sigmoid(
                        path_to_wav_cropped,
                        num_components,
                        out_dir=dir_out_abl,
                        is_plot=False,
                        annotation_file=path_to_annot,
                        ablation=abl,
                    )
        if args.parallel:
            p = Process(target=ablation_study_subprocess)
            p.start()
            processes.append(p)
        else:
            ablation_study_subprocess()

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
    for abl in ablations:
        dir_out_abl = os.path.join(args.dir_out, abl)
        results_ = gather_results(dir_out_abl, substitute_enst_dir=args.dir_enst)
        results[abl] = results_

        print('--------------------------')
        visualize_results(results_, results_, None, xlabel=f'{abl}', ylabel=f'{abl}')

