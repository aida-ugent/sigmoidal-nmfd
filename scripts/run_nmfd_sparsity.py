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
from .run_nmfd_vanilla import run_nmfd_vanilla


def run_nmfd_sparsity(*args, sparsity=1.0, do_warmup=False, **kwargs):
    return run_nmfd_vanilla(*args, **kwargs, sparsity=sparsity, do_warmup=do_warmup)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Apply NMFD with sparsity to a drum extract.')
    parser.add_argument('filepath', type=str,
                        help='Path to a .wav or .mp3 file of the audio to be decomposed.')
    parser.add_argument('num_components', type=int,
                        help='Number of NMFD components.')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--sparsity', type=float, default=1.0,
                        help='Sparsity hyperparameter.')
    parser.add_argument('--warmup', action='store_true',
                        help='Allow a few iterations of unconstrained optimization.')
    args = parser.parse_args()

    result = run_nmfd_sparsity(
        args.filepath,
        args.num_components,
        is_plot=args.plot,
        sparsity=args.sparsity,
        do_warmup=args.warmup,
        out_dir='.' if args.save else None,
    )
